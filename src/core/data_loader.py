from typing import Dict
import pandas as pd

def infer_schema_dict(df: pd.DataFrame, table_name: str) -> Dict:
    cols = []
    for c in df.columns:
        cols.append({"name": c, "dtype": str(df[c].dtype), "description": ""})
    return {"table_name": table_name, "columns": cols}