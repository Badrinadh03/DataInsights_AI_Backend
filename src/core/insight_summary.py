# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.core.llm_client import chat_complete, current_settings

def _basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    out["columns"] = list(map(str, df.columns))

    numeric = df.select_dtypes(include=["number"])
    top_numeric = {}
    if not numeric.empty:
        means = numeric.mean(numeric_only=True).sort_values(ascending=False).head(3)
        sums  = numeric.sum(numeric_only=True).sort_values(ascending=False).head(3)
        top_numeric["means"] = means.to_dict()
        top_numeric["sums"] = sums.to_dict()
    out["top_numeric"] = top_numeric

    # categorical quick take
    cat_cols = [c for c in df.columns if c not in numeric.columns]
    topcats = {}
    for c in cat_cols[:3]:
        try:
            vc = df[c].astype(str).value_counts(dropna=False).head(3)
            topcats[str(c)] = {str(k): int(v) for k, v in vc.items()}
        except Exception:
            continue
    out["top_categories"] = topcats

    return out

def _short_fallback(stats: Dict[str, Any], question: Optional[str]) -> str:
    rows = stats.get("shape", {}).get("rows", 0)
    cols = stats.get("shape", {}).get("cols", 0)
    cat = stats.get("top_categories", {})
    num = stats.get("top_numeric", {})
    pieces = [f"The result has {rows} rows × {cols} columns."]
    if num.get("sums"):
        k, v = next(iter(num["sums"].items()))
        pieces.append(f"Top aggregate column is '{k}'.")
    if cat:
        c0, vv = next(iter(cat.items()))
        vk = list(vv.keys())[0] if vv else None
        if vk:
            pieces.append(f"'{c0}' is dominated by '{vk}'.")
    if question:
        pieces.append(f"These findings relate to your question: {question[:120]}...")
    return " ".join(pieces[:3])

def summarize_dataframe(df: pd.DataFrame, question: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a concise 2–3 sentence summary focused on the user's question.
    Falls back to a short local summary if LLM is unavailable.
    """
    #stats = _basic_stats(df)
    print("Summarize DF called.")
    try:
        #cfg = current_settings()
        sys = "You summarize tabular query results concisely."
        user = f"""Write 2–3 crisp sentences that are DIRECTLY relevant to the user's question.
        Avoid generic stats; mention only the most important patterns or outliers.
        Keep the whole response under 400 characters.
        Also, use INR as the currency wherever applicable.

        Question: {question or ''}
        Columns: {list(map(str, df.columns))}
        Data: {df.to_dict(orient='records')}
"""
        print(user)
        text = chat_complete(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        ).strip()
        # Normalize whitespace and enforce hard limit
        text = " ".join(text.split())[:400]
        return {"summary": text}
    except Exception:
        print("Exception in Insights Summary")
        return {"summary": _short_fallback(stats, question), "stats": stats}
