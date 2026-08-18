from typing import List, Dict, Optional, Any, Tuple
import pandas as pd

from src.core.sanitize import sanitize_identifier, sanitize_columns
from src.core.type_coercion import coerce_numeric_like

from src.core.data_loader import infer_schema_dict
from src.core.llm_query_response import get_sql_query
from src.core.run_sql import run_sql_query
from src.core.chart_llm import make_llm_chart
from src.core.llm_client import current_settings
from src.core.insight_summary import summarize_dataframe

class InsightEngine:
    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or "data_"

    def prepare_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, str]:
        safe_df, mapping = sanitize_columns(df)
        safe_df = coerce_numeric_like(safe_df)
        safe_table = sanitize_identifier(self.table_name, is_table=True)
        return safe_df, mapping, safe_table

    def build_dictionary(self, df: pd.DataFrame, table_name: str) -> dict:
        d = infer_schema_dict(df, table_name)
        cols = [{"name": c, "dtype": str(df[c].dtype), "description": f"Column '{c}' of type {str(df[c].dtype)}."} for c in df.columns]
        d["columns"] = cols
        d["table_name"] = table_name
        return d

    def answer(self, question: str, df: pd.DataFrame, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        safe_df, mapping, safe_table = self.prepare_dataframe(df)
        data_dict = self.build_dictionary(safe_df, safe_table)

        cfg = current_settings()
        sql = get_sql_query(
            question=question,
            data_dictionary=data_dict,
            table_name=safe_table,
            history=history or [],
            provider=cfg["provider"],
            api_key=cfg["api_key"],
            model=cfg["model"],
        )

        result_df = run_sql_query(sql, safe_df, safe_table)

        fig = None
        try:
            fig = make_llm_chart(
                question, result_df,
                provider=cfg["provider"],
                api_key=cfg["api_key"],
                model=cfg["model"],
            )
        except Exception:
            fig = None

        insights = summarize_dataframe(result_df, question=question)

        return {
            "sql": sql,
            "result": result_df,
            "figure": fig,
            "insights": insights,
            "mapping": mapping,
            "table_name": safe_table,
        }