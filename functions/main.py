import os, sys, uuid, json, io, logging
from typing import Any, Dict, List
from collections import defaultdict, deque
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from dotenv import load_dotenv
load_dotenv(override=False)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- Your existing core modules ---
from src.services.insight_engine import InsightEngine
from src.core.llm_client import set_provider, current_settings, chat_complete
from src.core.sanitize import sanitize_identifier
from src.core.llm_query_response import get_sql_query
from src.core.run_sql import run_sql_query
from src.core.chart_llm import make_llm_chart  # still used by /v1/qa/answer
from src.core.insight_summary import summarize_dataframe
from src.core.qa_answers import get_fixed_answer
# ---- LLM provider from .env ----
set_provider(
    os.getenv("LLM_PROVIDER", "gemini"),
    os.getenv("MISTRAL_API_KEY", ""),
    os.getenv("OPENAI_API_KEY", ""),
    gemini_key=os.getenv("GEMINI_API_KEY", ""),
    model=(os.getenv("GEMINI_MODEL", "") or os.getenv("OPENAI_MODEL", "") or os.getenv("MISTRAL_MODEL", "") or "gemini-2.0-flash")
)

app = Flask(__name__)
raw_origins = (
    os.getenv("CORS_ORIGINS")
    or os.getenv("FRONTEND_URL")
    or os.getenv("FRONTEND_URLS")
    or "*"
)
origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
if not origins:
    origins = ["*"]
CORS(app, resources={r"/*": {"origins": origins}})
logging.basicConfig(level=logging.INFO)

# ---- In-memory stores (swap to disk if you want persistence) ----
DATASETS: Dict[str, pd.DataFrame] = {}
META: Dict[str, Dict[str, Any]] = {}

# recent uploads per client (max 5)
RECENT_UPLOADS = defaultdict(lambda: deque(maxlen=5))  # client_id -> deque([meta...])

# sessions: client_id -> { dataset_id: { session_id: session_obj } }
# session_obj = {session_id, dataset_id, name, created_at, updated_at, messages[...]}
SESSIONS = defaultdict(lambda: defaultdict(dict))


# ---------------- utils ----------------
def _json_error(message, code=500, kind="internal", detail=None):
    logging.error("%s | kind=%s | detail=%s", message, kind, detail)
    return jsonify({"error": message, "kind": kind, "detail": detail}), code

def _safe_table_name(ds_id: str, filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename or "data"))[0]
    raw = f"{base}_{ds_id[:8]}"
    return sanitize_identifier(raw, is_table=True)

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _load_schema_sheet(meta: Dict[str, Any], sheet_name: str):
    """
    Load a schema/data-dictionary sheet from the same Excel file.
    Expected columns (case-insensitive): name, dtype, description
    - 'name' will be sanitized to match the column names used in SQL.
    """
    if not meta or meta["type"] != "excel":
        return None
    xls = pd.ExcelFile(io.BytesIO(meta["bytes"]))
    if sheet_name not in xls.sheet_names:
        return None

    schema_df = xls.parse(sheet_name)
    schema_df.columns = [str(c).strip().lower() for c in schema_df.columns]
    if "name" not in schema_df.columns:
        return None

    cols = []
    for _, row in schema_df.iterrows():
        raw_nm = str(row.get("name", "")).strip()
        if not raw_nm:
            continue
        try:
            nm = sanitize_identifier(raw_nm, is_table=False)
        except TypeError:
            nm = sanitize_identifier(raw_nm)
        dt = str(row.get("dtype", "") or "object").strip()
        desc = str(row.get("description", "") or "").strip()
        cols.append({"name": nm, "dtype": dt, "description": desc})
    return cols


# ---------------- health ----------------
@app.get("/health")
def health():
    cfg = current_settings()
    return jsonify({"status": "ok", "provider": cfg["provider"], "model": cfg["model"]}), 200


# ---------------- datasets ----------------
@app.post("/v1/datasets")
def create_dataset():
    if "file" not in request.files:
        return _json_error("file field is required", 400, "validation")

    client_id = request.form.get("client_id") or request.headers.get("X-Client-Id") or ""
    f = request.files["file"]
    name = f.filename or "data.csv"
    raw = f.read()

    try:
        if name.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
            ftype = "csv"
        else:
            df = pd.read_excel(io.BytesIO(raw))
            ftype = "excel"
    except Exception as e:
        return _json_error(f"failed to read file: {e}", 400, "ingestion")

    ds_id = uuid.uuid4().hex
    table_name = _safe_table_name(ds_id, name)
    created_at = _now_iso()

    DATASETS[ds_id] = df
    META[ds_id] = {
        "name": name, "bytes": raw, "type": ftype,
        "table_name": table_name, "created_at": created_at
    }

    if client_id:
        RECENT_UPLOADS[client_id].appendleft({
            "dataset_id": ds_id, "name": name, "rows": int(df.shape[0]),
            "cols": int(df.shape[1]), "table_name": table_name, "created_at": created_at
        })

    return jsonify({
        "dataset_id": ds_id,
        "name": name,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "table_name": table_name,
        "created_at": created_at,
    }), 201

@app.get("/v1/datasets/<ds_id>/preview")
def preview_dataset(ds_id: str):
    n = int(request.args.get("n", 50))
    df = DATASETS.get(ds_id)
    if df is None:
        return _json_error("dataset not found", 404, "not_found")
    prev = df.head(n).to_dict(orient="records")
    return jsonify({"rows": prev, "columns": list(map(str, df.columns))})

@app.get("/v1/datasets/recent")
def recent_datasets():
    client_id = request.args.get("client_id", "")
    limit = int(request.args.get("limit", 5))
    items = list(RECENT_UPLOADS.get(client_id, []))[:limit] if client_id else []
    return jsonify({"items": items})

@app.get("/v1/datasets/<ds_id>/schema")
def get_effective_schema(ds_id: str):
    """
    Return the effective LLM schema context for a dataset:
      - sanitized table name
      - sanitized column list (with dtype/description merged from an Excel schema sheet if provided)
      - original->sanitized mapping
    Query params:
      - schema_sheet=<sheet_name>  (optional, Excel only)
    """
    schema_sheet = (request.args.get("schema_sheet") or "").strip()

    df = DATASETS.get(ds_id)
    meta = META.get(ds_id)
    if df is None or not meta:
        return _json_error("dataset not found", 404, "not_found")

    engine = InsightEngine(table_name=meta["table_name"])
    safe_df, mapping, safe_table = engine.prepare_dataframe(df)
    data_dict = engine.build_dictionary(safe_df, safe_table)

    if schema_sheet:
        sheet_cols = _load_schema_sheet(meta, schema_sheet)
        if sheet_cols:
            sheet_by_name = {c["name"]: c for c in sheet_cols if c.get("name")}
            merged = []
            for col_name in map(str, safe_df.columns):
                base = {"name": col_name}
                sc = sheet_by_name.get(col_name)
                if sc:
                    if sc.get("dtype"):       base["dtype"] = sc["dtype"]
                    if sc.get("description"): base["description"] = sc["description"]
                merged.append(base)
            data_dict["columns"] = merged

    return jsonify({
        "dataset_id": ds_id,
        "table_name": safe_table,
        "columns": data_dict.get("columns", []),
        "mapping": mapping,
        "rows": int(safe_df.shape[0]),
        "cols": int(safe_df.shape[1]),
        "schema_sheet_applied": bool(schema_sheet),
        "schema_sheet": schema_sheet or None,
    }), 200


# ---------------- chat sessions ----------------
@app.post("/v1/chats/session")
def create_session():
    body = request.get_json(force=True, silent=True) or {}
    client_id = body.get("client_id", "")
    dataset_id = body.get("dataset_id", "")
    name = (body.get("name") or "New chat").strip()
    if not client_id or not dataset_id:
        return _json_error("client_id and dataset_id required", 400, "validation")
    sid = uuid.uuid4().hex
    now = _now_iso()
    sess = {"session_id": sid, "dataset_id": dataset_id, "name": name,
            "created_at": now, "updated_at": now, "messages": []}
    SESSIONS[client_id][dataset_id][sid] = sess
    return jsonify({"session": {k: v for k, v in sess.items() if k != "messages"}}), 201

@app.get("/v1/chats/sessions")
def list_sessions():
    client_id = request.args.get("client_id", "")
    dataset_id = request.args.get("dataset_id", "")
    if not client_id or not dataset_id:
        return _json_error("client_id and dataset_id required", 400, "validation")
    sessions = list(SESSIONS.get(client_id, {}).get(dataset_id, {}).values())
    out = []
    for s in sessions:
        out.append({
            "session_id": s["session_id"],
            "name": s["name"],
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
            "message_count": len(s["messages"]),
        })
    out.sort(key=lambda x: x["updated_at"], reverse=True)
    return jsonify({"sessions": out})

@app.get("/v1/chats/<session_id>")
def get_session(session_id: str):
    client_id = request.args.get("client_id", "")
    dataset_id = request.args.get("dataset_id", "")
    if not client_id or not dataset_id:
        return _json_error("client_id and dataset_id required", 400, "validation")
    sess = SESSIONS.get(client_id, {}).get(dataset_id, {}).get(session_id)
    if not sess:
        return _json_error("session not found", 404, "not_found")
    return jsonify({"session": {k: v for k, v in sess.items() if k != "messages"},
                    "messages": sess["messages"]})

@app.patch("/v1/chats/<session_id>")
def rename_session(session_id: str):
    body = request.get_json(force=True, silent=True) or {}
    client_id = body.get("client_id", "")
    dataset_id = body.get("dataset_id", "")
    new_name = (body.get("name") or "").strip()
    if not client_id or not dataset_id or not new_name:
        return _json_error("client_id, dataset_id and name required", 400, "validation")
    sess = SESSIONS.get(client_id, {}).get(dataset_id, {}).get(session_id)
    if not sess:
        return _json_error("session not found", 404, "not_found")
    sess["name"] = new_name
    sess["updated_at"] = _now_iso()
    return jsonify({"ok": True})


# ---------------- QA/answer ----------------
@app.post("/v1/qa/answer")
def qa_answer():
    body = request.get_json(force=True, silent=True) or {}
    ds_id = body.get("dataset_id")
    question = (body.get("question") or "").strip()
    history = body.get("history") or []
    schema_sheet = (body.get("schema_sheet") or "").strip()

    if not ds_id or not question:
        return _json_error("dataset_id and question are required", 400, "validation")

    df = DATASETS.get(ds_id)
    meta = META.get(ds_id)
    if df is None or not meta:
        return _json_error("dataset not found", 404, "not_found")

    # --- FIXED RESPONSE BYPASS (no SQL/LLM/DF prep) ---
    fixed = get_fixed_answer(question)
    if fixed:
        # Persist session (kept as close to your logic as possible)
        client_id = body.get("client_id") or request.headers.get("X-Client-Id") or ""
        session_id = (body.get("session_id") or "").strip()
        session_name = None

        if client_id:
            # Use setdefault here so we don't KeyError if buckets don't exist yet
            client_bucket = SESSIONS.setdefault(client_id, {})
            ds_bucket = client_bucket.setdefault(ds_id, {})

            if not session_id:
                session_id = uuid.uuid4().hex
                ds_bucket[session_id] = {
                    "session_id": session_id,
                    "dataset_id": ds_id,
                    "name": "New chat",
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "messages": [],
                }

            sess = ds_bucket.get(session_id)
            if not sess:
                session_id = uuid.uuid4().hex
                sess = {
                    "session_id": session_id,
                    "dataset_id": ds_id,
                    "name": "New chat",
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "messages": [],
                }
                ds_bucket[session_id] = sess

            if sess.get("name", "").strip().lower() in {"new chat", "newchat", "new_chat"} and len(sess.get("messages", [])) == 0:
                sess["name"] = (question[:50] + "…") if len(question) > 50 else (question or "Chat")

            now = _now_iso()
            sess["messages"].append({"role": "user", "content": question, "ts": now})
            sess["messages"].append({"role": "assistant", "content": fixed, "ts": now, "fixed": True})
            sess["updated_at"] = now
            session_name = sess["name"]

        return jsonify({
            "sql": "",  # nothing executed
            "result": {"rows": [], "columns": []},
            "chart": None,
            "table_name": meta.get("table_name"),  # still useful context without DF prep
            "summary": fixed,                       # your UI already reads 'summary'
            "session_id": session_id or None,
            "session_name": session_name,
            "fixed": True,                          # optional flag for the UI
            "note": "canned_response",              # optional hint
        })

    # -------- YOUR ORIGINAL LOGIC CONTINUES BELOW --------

    engine = InsightEngine(table_name=meta["table_name"])
    safe_df, mapping, safe_table = engine.prepare_dataframe(df)
    data_dict = engine.build_dictionary(safe_df, safe_table)
    schema_sheet = "Sheet2"
    if schema_sheet:
        print("Inside Schema")
        cols = _load_schema_sheet(meta, schema_sheet)
        if cols:
            data_dict["columns"] = cols
        print(data_dict)
    cfg = current_settings()

    # 1) SQL generation
    try:
        sql = get_sql_query(
            question=question,
            data_dictionary=data_dict,
            table_name=safe_table,
            history=history,
            provider=cfg["provider"],
            api_key=cfg["api_key"],
            model=cfg["model"],
        )
    except Exception as e:
        return _json_error(f"LLM SQL generation failed: {e}", 502, "llm_error")

    # 2) Run SQL
    try:
        result_df = run_sql_query(sql, safe_df, safe_table)
    except Exception as e:
        return _json_error(f"SQL execution failed: {e}", 400, "sql_error", {"sql": sql})

    # 3) Chart (best-effort)
    fig = None
    try:
        from plotly.io import to_json as plotly_to_json
        fig_obj = make_llm_chart(
            question, result_df,
            provider=cfg["provider"],
            api_key=cfg["api_key"], 
            model=cfg["model"],
        )
        fig = json.loads(plotly_to_json(fig_obj)) if fig_obj is not None else None
    except Exception:
        fig = None

    # 4) Summary (short)
    try:
        summary = summarize_dataframe(result_df, question=question).get("summary", "")
    except Exception:
        summary = ""

    # 5) Persist session (auto-create if needed)
    client_id = body.get("client_id") or request.headers.get("X-Client-Id") or ""
    session_id = (body.get("session_id") or "").strip()
    session_name = None

    if client_id:
        ds_bucket = SESSIONS[client_id][ds_id]

        if not session_id:
            session_id = uuid.uuid4().hex
            ds_bucket[session_id] = {
                "session_id": session_id,
                "dataset_id": ds_id,
                "name": "New chat",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "messages": [],
            }

        sess = ds_bucket[session_id]

        if sess.get("name", "").strip().lower() in {"new chat", "newchat", "new_chat"} and len(sess.get("messages", [])) == 0:
            sess["name"] = (question[:50] + "…") if len(question) > 50 else (question or "Chat")

        now = _now_iso()
        sess["messages"].append({"role": "user", "content": question, "ts": now})
        sess["messages"].append({"role": "assistant", "content": summary, "ts": now, "sql": sql})
        sess["updated_at"] = now
        session_name = sess["name"]

    return jsonify({
        "sql": sql,
        "result": {
            "rows": result_df.to_dict(orient="records"),
            "columns": list(map(str, result_df.columns)),
        },
        "chart": fig,
        "table_name": safe_table,
        "summary": summary,
        "session_id": session_id or None,
        "session_name": session_name,
    })



# ---------- LLM helper for Auto-Insights ----------
def _extract_column_names_for_llm(data_dict: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    """
    Prefer the sanitized dictionary columns (with 'name'), else fall back to df columns.
    """
    try:
        cols = data_dict.get("columns") or []
        names = []
        for c in cols:
            if isinstance(c, dict) and c.get("name"):
                names.append(str(c["name"]).strip())
            else:
                names.append(str(c).strip())
        names = [n for n in names if n]
        if names:
            return names
    except Exception:
        pass
    return [str(c) for c in df.columns]

def _chat_hypotheses(columns: List[str], k: int, cfg: Dict[str, Any]) -> List[str]:
    """
    Use the configured LLM provider to emit a JSON array of short, schema-aware questions.
    """
    api_key = cfg.get("api_key", "")
    model = (cfg.get("model") or "").strip()
    if not api_key:
        raise RuntimeError(f"{cfg.get('provider', 'LLM').upper()} API key is not configured")

    system = (
        "You are a sharp data analyst. Given a list of lower_snake_case columns from a single table, "
        "propose SHORT, high-value, domain-agnostic questions that can be answered from that table. "
        "Vary aggregates (sum/avg/count), rankings, and comparisons across categories; include trends only if date-like columns exist. "
        "Return ONLY a JSON array of strings. No prose, no code fences."
    )
    user = (
        f"Columns: {columns}\n"
        f"Produce up to {max(1, min(int(k), 12))} distinct questions. "
        f"Each must be a single sentence, under 100 characters. "
        f"Output JSON array only."
    )

    content = chat_complete(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        provider=cfg.get("provider"),
        api_key=api_key,
        model=model,
        temperature=0.9,
    )

    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    try:
        arr = json.loads(text)
        out = []
        for x in arr:
            q = str(x).strip()
            if not q.endswith("?"):
                q += "?"
            if q:
                out.append(q)
        return out[:k] if out else []
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM hypotheses JSON: {e}")


# ---------------- insights/auto (no charts) ----------------
@app.post("/v1/insights/auto")
def auto_insights():
    """
    Generate concise, LLM-guided insights:
    1) Hypothesis generation (configured LLM, JSON array of questions)
    2) SQL generation (reuses get_sql_query)
    3) Execute on sanitized DataFrame
    4) Summarize (few sentences)
    """
    body = request.get_json(force=True, silent=True) or {}
    ds_id = (body.get("dataset_id") or "").strip()
    k = int(body.get("k", 5))
    schema_sheet = (body.get("schema_sheet") or "").strip()

    if not ds_id:
        return _json_error("dataset_id is required", 400, "validation")

    df = DATASETS.get(ds_id)
    meta = META.get(ds_id)
    if df is None or not meta:
        return _json_error("dataset not found", 404, "not_found")

    engine = InsightEngine(table_name=meta["table_name"])
    safe_df, mapping, safe_table = engine.prepare_dataframe(df)
    data_dict = engine.build_dictionary(safe_df, safe_table)

    if schema_sheet:
        cols = _load_schema_sheet(meta, schema_sheet)
        if cols:
            data_dict["columns"] = cols

    cfg = current_settings()

    # 1) Hypotheses
    try:
        colnames = _extract_column_names_for_llm(data_dict, safe_df)
        questions = _chat_hypotheses(colnames, k, cfg)
        if not questions:
            return jsonify({"dataset_id": ds_id, "items": []}), 200
    except Exception as e:
        return _json_error(f"LLM hypothesis generation failed: {e}", 502, "llm_error")

    items = []
    for q in questions:
        try:
            # 2) SQL generation
            sql = get_sql_query(
                question=q,
                data_dictionary=data_dict,
                table_name=safe_table,
                history=[],
                provider=cfg["provider"],
                api_key=cfg["api_key"],
                model=cfg["model"],
            )

            # 3) Execute
            res_df = run_sql_query(sql, safe_df, safe_table)

            # 4) Summarize (few sentences)
            try:
                print("Trying summary")
                summary = summarize_dataframe(res_df, question=q).get("summary", "")
                print("Getting the summary")
                print(summary)
            except Exception as e:
                print(e)
                print("You hit the exception bro")
                summary = ""

            items.append({
                "question": q,
                "sql": sql,
                "summary": summary,
                "result_preview": res_df.head(50).to_dict(orient="records"),
            })

        except Exception as e:
            items.append({
                "question": q,
                "sql": "",
                "summary": f"(skipped) {e}",
                "result_preview": [],
            })
    num_ques = max(k, 5)
    questions = _chat_hypotheses(colnames, num_ques, cfg)
    print(questions)

    return jsonify({"dataset_id": ds_id, "items": items, "suggested_questions" : questions}), 200
    


# ---------------- main ----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
