import plotly.express as px
import pandas as pd

def build_performance_charts(gb_sum: pd.DataFrame):
    out = {}
    out["ctd_visit"] = px.line(gb_sum, x='month_year', y='ctd_visit', color='employee_name', markers=True, title='Total Visit per Month')
    out["avg_distance"] = px.line(gb_sum, x='month_year', y='avg_distance_km', color='employee_name', markers=True, title='Average Distance per Visit (km)')
    out["avg_walk_time"] = px.line(gb_sum, x='month_year', y='avg_time_between_minute', color='employee_name', markers=True, title='Average Time Between Visits (min)')
    out["avg_speed"] = px.line(gb_sum, x='month_year', y='avg_speed_kmpm', color='employee_name', markers=True, title='Median Speed (km/min)')
    return out
