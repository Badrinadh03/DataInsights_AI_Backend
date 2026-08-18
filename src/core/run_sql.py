import re
import sqlite3
import pandas as pd

_SQL_FENCE = re.compile(r"```(?:\s*sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

def _normalize_sql(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    m = _SQL_FENCE.search(t)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"^\s*sql\s*[:\-]*\s*\n", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*sql\s+", "", t, flags=re.IGNORECASE)
    parts = [p.strip() for p in t.split(";") if p.strip()]
    return parts[0] if parts else ""

def _force_single_table(sql: str, table_name: str) -> str:
    """Ensure the query references our in-memory table; rewrite the first FROM."""
    if not sql:
        return sql
    sql = re.sub(r'(?i)\bfrom\s+("?[A-Za-z_][A-Za-z0-9_]*"?)',
                 f'FROM "{table_name}"', sql, count=1)
    return sql

def run_sql_query(sql: str, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Load df into an in-memory SQLite table and execute the given SQL."""
    sql = _normalize_sql(sql)
    sql = _force_single_table(sql, table_name)
    if not sql:
        raise ValueError("SQL is empty")
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        res = pd.read_sql_query(sql, conn)
        return res
    finally:
        conn.close()