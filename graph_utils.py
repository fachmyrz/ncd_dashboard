import plotly.express as px
from data_preprocess import get_summary_data, df_visit

summary, data = get_summary_data()
gb_sum = data.groupby(["month_year","employee_name"], as_index=False).agg({"ctd_visit":"sum","avg_distance_km":"mean","avg_time_between_minute":"mean","avg_speed_kmpm":"median"})
ctd_visit = px.line(gb_sum, x="month_year", y="ctd_visit", color="employee_name", markers=True)
avg_distance = px.line(gb_sum, x="month_year", y="avg_distance_km", color="employee_name", markers=True)
avg_walk_time = px.line(gb_sum, x="month_year", y="avg_time_between_minute", color="employee_name", markers=True)
avg_speed = px.line(gb_sum, x="month_year", y="avg_speed_kmpm", color="employee_name", markers=True)
