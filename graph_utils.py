import plotly.express as px
import pandas as pd
from data_preprocess import df_visit

def performance_lines():
    if df_visit is None or df_visit.empty:
        return None, None, None, None
    v = df_visit.copy()
    v["date"] = pd.to_datetime(v["visit_datetime"], errors="coerce").dt.date
    if v["date"].isna().all():
        return None, None, None, None
    v["month_year"] = pd.to_datetime(v["date"]).astype("datetime64[M]").astype(str)
    gb = v.groupby(["month_year","employee_name"]).agg(
        ctd_visit=("date","count")
    ).reset_index()
    ctd_visit = px.line(gb, x="month_year", y="ctd_visit", color="employee_name", markers=True)
    return ctd_visit, None, None, None
