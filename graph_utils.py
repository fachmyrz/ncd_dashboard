import pandas as pd
import plotly.express as px
from data_preprocess import visit_metrics, revenue_monthly

def visits_by_employee():
    if "last_visited_by" in visit_metrics.columns and not visit_metrics.empty:
        tmp = visit_metrics.groupby("last_visited_by", dropna=True).agg(
            avg_weekly=("avg_weekly_visits","mean"),
            visits_90=("visits_last_90","sum")
        ).reset_index()
        fig1 = px.bar(tmp.sort_values("visits_90", ascending=False), x="last_visited_by", y="visits_90")
        fig2 = px.bar(tmp.sort_values("avg_weekly", ascending=False), x="last_visited_by", y="avg_weekly")
        return fig1, fig2
    return None, None

def revenue_trend_overall():
    if not revenue_monthly.empty:
        agg = revenue_monthly.groupby("month")["monthly_revenue"].sum().reset_index()
        return px.line(agg, x="month", y="monthly_revenue", markers=True)
    return None
