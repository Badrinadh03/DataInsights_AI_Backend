import re
import pandas as pd
import numpy as np

_NUMERIC_CHARS = re.compile(r"[^\d.\-\(\)]")

def _to_number(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ("", "-", "–", "—"):
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    # strip everything except digits . - (we already removed parens)
    s = _NUMERIC_CHARS.sub("", s.replace(",", ""))
    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        x = float(s)
    except Exception:
        return np.nan
    return -x if neg else x

def coerce_numeric_like(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Convert object columns that look mostly numeric (with commas, currency symbols, parens) to floats."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            ser = out[c].astype(str)
            # try converting
            conv = ser.map(_to_number)
            # if enough values became numbers, accept conversion
            nonnull = conv.notna().sum()
            if len(ser) > 0 and (nonnull / len(ser)) >= threshold:
                out[c] = conv
    return out
