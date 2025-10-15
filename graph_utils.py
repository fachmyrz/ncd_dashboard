import plotly.express as px
import pandas as pd
from data_preprocess import avail_df_merge, df_visits, revenue_monthly

def fig_engagement_by_bde():
    if df_visits is None or df_visits.empty:
        return None
    if "employee_name" not in df_visits.columns or "visit_datetime" not in df_visits.columns:
        return None
    df = df_visits.copy()
    df["month"] = df["visit_datetime"].dt.to_period("M").astype(str)
    gb = df.groupby(["employee_name","month"]).size().reset_index(name="visits")
    fig = px.line(gb, x="month", y="visits", color="employee_name", markers=True)
    fig.update_layout(title_text="Monthly Visits per BDE", xaxis_title="Month", yaxis_title="Visits")
    return fig

def fig_activity_distribution():
    df = avail_df_merge.copy()
    if df.empty or "tag" not in df.columns:
        return None
    cnt = df["tag"].value_counts().reset_index().rename(columns={"index":"Activity","tag":"Count"})
    fig = px.pie(cnt, values="Count", names="Activity", title="Dealer Activity Distribution")
    return fig

def fig_top_revenue_dealers(n=10):
    df = avail_df_merge.copy()
    if df.empty or "avg_monthly_revenue" not in df.columns:
        return None
    tmp = df.dropna(subset=["avg_monthly_revenue"]).sort_values("avg_monthly_revenue", ascending=False).drop_duplicates(subset=["id_dealer_outlet"])
    tmp = tmp.head(n)
    if tmp.empty:
        return None
    fig = px.bar(tmp, x="dealer_name" if "dealer_name" in tmp.columns else "client_name", y="avg_monthly_revenue", hover_data=["brand","city"], title=f"Top {n} Dealers by Avg Monthly Revenue")
    return fig
