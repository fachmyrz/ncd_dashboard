# graph_utils.py
# lightweight: uses precomputed summary from data_preprocess.compute_all
import plotly.express as px
from data_preprocess import compute_all

computed = compute_all()
summary, data = computed.get("sum_df"), computed.get("df_visits")
if summary is not None and not summary.empty:
    gb_sum = summary.groupby(["month_year","employee_name"], as_index=False).agg({"ctd_visit":"sum","avg_distance_km":"mean"})
    ctd_visit = px.line(gb_sum, x="month_year", y="ctd_visit", color="employee_name", markers=True)
    ctd_visit.update_layout(title_text="Total Visit per Bulan")
    ctd_visit.show()
