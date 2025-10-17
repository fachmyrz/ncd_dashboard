# graph_utils.py
import plotly.express as px
from data_preprocess import compute_all
import pandas as pd

computed = compute_all()
visits = computed.get("visits", pd.DataFrame())
if not visits.empty:
    v = visits.copy()
    if "visit_datetime" in v.columns:
        v["date"] = pd.to_datetime(v["visit_datetime"], errors="coerce").dt.date
        v = v.dropna(subset=["date"])
        v["month_year"] = pd.to_datetime(v["date"]).dt.to_period("M").astype(str)
        gb = v.groupby(["month_year","employee_name"], as_index=False).size().rename(columns={"size":"ctd_visit"})
        fig = px.line(gb, x="month_year", y="ctd_visit", color="employee_name", markers=True)
        fig.update_layout(title_text="Total Visit per Month")
        fig.show()
