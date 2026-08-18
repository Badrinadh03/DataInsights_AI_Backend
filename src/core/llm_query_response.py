import re
from typing import List, Dict, Optional
from src.core.llm_client import chat_complete, current_settings

# Extract fenced ```sql blocks or stray leading 'sql' labels
_SQL_FENCE = re.compile(r"```(?:\s*sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

def _extract_sql_block(text: str) -> str:
    """Normalize LLM output to a single SQL statement."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    m = _SQL_FENCE.search(t)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"^\s*sql\s*[:\-]*\s*\n", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*sql\s+", "", t, flags=re.IGNORECASE)
    parts = [p.strip() for p in t.split(";") if p.strip()]
    return parts[0] if parts else ""

def _build_sql_prompt(
    question: str,
    data_dictionary: dict,
    table_name: str,
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    # Build a readable schema block of sanitized columns
    cols = data_dictionary.get("columns", [])
    lines = []
    for c in cols:
        nm = str(c.get("name", "")).strip()
        dt = str(c.get("dtype", "")).strip()
        desc = str(c.get("description", "") or "").strip()
        if desc:
            lines.append(f"- {nm} ({dt}): {desc}")
        else:
            lines.append(f"- {nm} ({dt})")
    cols_block = "\n".join(lines)

    # Include a small rolling history to help disambiguate
    hist_text = ""
    if history:
        trimmed = history[-6:]
        hist_lines = []
        for h in trimmed:
            role = (h.get("role") or "user").strip()
            content = (h.get("content") or "").replace("\n", " ").strip()
            hist_lines.append(f"{role}: {content}")
        hist_text = "\n".join(hist_lines)

    # Prompt with explicit, relevant SQLite guidelines
    prompt = f"""You are a precise SQL generator for SQLite.

You must produce ONE valid SQL statement (you may use a WITH/CTE if helpful).
Return ONLY the SQL query; do not add explanations or code fences.

## Table (use ONLY this table and these sanitized columns)
Table: {table_name}
Columns:
{cols_block}

## Conversation context (optional)
{hist_text}

## Question
{question}

## Guidelines
- Use ONLY the sanitized lower_snake_case table/column names listed above. Do not invent names.
- Use double quotes for identifiers when needed (e.g., "product_type").
- Prefer case-insensitive text comparisons: either use LOWER(col) and lowercase literals, or `COLLATE NOCASE`.
- Use LIKE with wildcards for fuzzy matching; users may not provide exact names.
- Use table aliases (e.g., `{table_name} AS t`) in the SQL clauses for clarity.
- Prefer GROUP BY + ORDER BY for rankings or aggregations.
- Give meaningful aliases to output columns with AS (e.g., AS total_unrealised_gain_loss).
- Avoid SELECT *; list explicit columns.
- Assume numeric columns are already numeric (no thousands separators); avoid unnecessary REPLACE().
- Ensure deterministic results: when ranking or returning top-k, ORDER BY explicit columns and use LIMIT k.
- Use window functions for top-N per group (e.g., ROW_NUMBER() OVER (PARTITION BY grp ORDER BY metric DESC)).
- NULL-safety: COALESCE(col, 0) for aggregates; divide as 1.0 * num / NULLIF(den, 0).
- Dates: if ISO-like strings, you may use DATE()/STRFTIME(); otherwise avoid parsing and prefer substring filters.
- Text hygiene: TRIM/LOWER before comparisons to handle stray spaces/case.
- Booleans in text: LOWER(col) IN ('y','yes','true','1') when needed.
- Unique counts: COUNT(DISTINCT col) for “unique”.
- Use CASE to derive labeled buckets; keep aliases descriptive.
- SQLite portability: do NOT use REGEXP, ILIKE, FULL OUTER JOIN, or non-SQLite functions.
- One statement only (CTEs allowed). No comments, no DDL.
- For Numerical values, lets round it off to two decimals

Return ONLY the SQL query, with no surrounding ``` fences and no 'sql' prefix.
"""
    return [{"role": "user", "content": prompt}]

def get_sql_query(
    question: str,
    data_dictionary: dict,
    table_name: str,
    history: Optional[List[Dict[str, str]]] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs
) -> str:
    cfg = current_settings()
    messages = _build_sql_prompt(question, data_dictionary, table_name, history)
    text = chat_complete(
        messages,
        provider=provider or cfg["provider"],
        api_key=api_key or cfg["api_key"],
        model=model or cfg["model"],
        temperature=temperature,
        **kwargs
    )
    return _extract_sql_block(text)
