import plotly.express as px
from data_preprocess import avail_df_merge, df_visits

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
    if df is None or df.empty or "tag" not in df.columns:
        return None
    cnt = df["tag"].value_counts().reset_index().rename(columns={"index":"Activity","tag":"Count"})
    fig = px.pie(cnt, values="Count", names="Activity", title="Dealer Activity Distribution")
    return fig

def fig_top_dealers_by_visits(n=10):
    df = avail_df_merge.copy()
    if df is None or df.empty or "visits_last_N" not in df.columns:
        return None
    tmp = df.drop_duplicates(subset=["id_dealer_outlet"]).sort_values("visits_last_N", ascending=False).head(n)
    if tmp.empty:
        return None
    fig = px.bar(tmp, x="client_name", y="visits_last_N", hover_data=["brand","city"], title=f"Top {n} Dealers by Visits (last 90 days)")
    return fig
