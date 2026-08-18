import re
import pandas as pd

def sanitize_identifier(name: str, is_table: bool = False) -> str:
    """Lowercase snake_case; ensure starts with letter; keep a-z0-9_ only."""
    if not isinstance(name, str):
        name = str(name)
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    if not n:
        n = "table" if is_table else "col"
    if n[0].isdigit():
        n = ("t_" if is_table else "c_") + n
    return n

def _unique_names(names):
    seen = {}
    out = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

def sanitize_columns(df: pd.DataFrame):
    """Return (df_with_sanitized_cols, mapping_original_to_sanitized)."""
    mapping = {}
    new_cols = []
    for c in df.columns:
        sc = sanitize_identifier(str(c), is_table=False)
        mapping[str(c)] = sc
        new_cols.append(sc)
    # ensure uniqueness
    new_cols = _unique_names(new_cols)
    out = df.copy()
    out.columns = new_cols
    return out, mapping