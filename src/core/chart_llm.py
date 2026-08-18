from typing import Optional, List, Dict
import re
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.core.llm_client import chat_complete, current_settings

CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)

def _build_chart_prompt(question: str, df: pd.DataFrame, data_dictionary: dict) -> List[Dict[str, str]]:
    # Build a readable schema block of sanitized columns from data dictionary
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

    # Sample rows for context (this can also be helpful to the model)
    sample = df.head(5).to_dict(orient="records")

    # The main prompt for chart generation
    prompt = f"""You are a Plotly chart code generator in Python.

## Goal
Given the pandas DataFrame `df` with columns below and a question, generate **one** clear, concise chart as executable Python code that assigns a Plotly figure to a variable named **`fig`**.

## Chart Selection Heuristics
- Choose the simplest chart that answers the question.
- Use time series for datetime + numeric columns.
- For categories vs numeric data, use a bar chart.
- If two numeric columns are involved, use a scatter plot.
- For numeric distribution, use a histogram.
- For part-to-whole charts (categories ≤ 8), use a pie chart.

## Data Schema (for your reasoning)
Columns:
{cols_block}

Sample rows (first 5):
{sample}

## Question:
{question}

## Guidelines:
- Return only the Python code that creates the `fig` variable (no explanations, no additional output).
- Do not import libraries (they are pre-imported).
- Use clear axis labels and titles.
- Keep styling basic and minimal; avoid custom themes or colors.
- Ensure numeric columns are properly aggregated and categorized.
- Handle missing values as necessary.

"""
    return [{"role": "user", "content": prompt}]


def _auto_chart(df: pd.DataFrame):
    """Heuristic fallback chart if LLM code fails or is empty."""
    if df is None or df.empty:
        return None
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]

    if dt_cols and num_cols:
        x = dt_cols[0]; y = num_cols[0]
        try:
            return px.line(df.sort_values(x), x=x, y=y, title=f"{y} over {x}")
        except Exception:
            pass

    if cat_cols and num_cols:
        x = cat_cols[0]; y = num_cols[0]
        try:
            agg = df.groupby(x, dropna=False)[y].sum().reset_index().sort_values(y, ascending=False).head(20)
            return px.bar(agg, x=x, y=y, title=f"{y} by {x}")
        except Exception:
            pass

    if num_cols:
        try:
            return px.histogram(df, x=num_cols[0], nbins=30, title=f"Distribution of {num_cols[0]}")
        except Exception:
            pass

    try:
        numeric = df.select_dtypes("number")
        if not numeric.empty:
            return px.imshow(numeric.head(20).T, title="Numeric snapshot")
    except Exception:
        pass
    return None

def make_llm_chart(question: str,
                   result_df: pd.DataFrame,
                   provider: Optional[str] = None,
                   api_key: Optional[str] = None,
                   model: Optional[str] = None,
                   temperature: float = 0.1,
                   **kwargs):
    if result_df is None or result_df.empty:
        return None
    cfg = current_settings()
    msgs = _build_chart_prompt(question, result_df)
    code = chat_complete(msgs,
                         provider=provider or cfg["provider"],
                         api_key=api_key or cfg["api_key"],
                         model=model or cfg["model"],
                         temperature=temperature,
                         **kwargs)
    src = _extract_code(code)

    safe_globals = {
        "__builtins__": {"len": len, "min": min, "max": max, "sum": sum, "range": range},
        "px": px,
        "go": go,
        "pd": pd,
    }
    safe_locals = {"df": result_df.copy()}
    try:
        exec(src, safe_globals, safe_locals)
        fig = safe_locals.get("fig", None)
        if fig is not None:
            return fig
    except Exception:
        pass
    return _auto_chart(result_df)