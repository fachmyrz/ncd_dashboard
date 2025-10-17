import plotly.express as px
from data_preprocess import *

summary,data = get_summary_data()

if not data.empty:
    gb_sum = data.groupby(['month_year','employee_name']).agg({'ctd_visit':'sum','avg_distance_km':'mean','avg_time_between_minute':'mean','avg_speed_kmpm':'median'}).reset_index()
    ctd_visit = px.line(gb_sum,x='month_year',y='ctd_visit',color='employee_name',markers=True)
    avg_distance = px.line(gb_sum,x='month_year',y='avg_distance_km',color='employee_name',markers=True)
    avg_walk_time = px.line(gb_sum,x='month_year',y='avg_time_between_minute',color='employee_name',markers=True)
    avg_speed = px.line(gb_sum,x='month_year',y='avg_speed_kmpm',color='employee_name',markers=True)
    ctd_visit.update_layout(title_text='Total Visit per Bulan')
    avg_distance.update_layout(title_text='Rata-rata Jarak per Visit')
    avg_walk_time.update_layout(title_text='Rata-rata Waktu antar Visit')
    avg_speed.update_layout(title_text='Rata-rata Kecepatan antar Visit')
    ctd_visit.show()
    avg_distance.show()
    avg_walk_time.show()
    avg_speed.show()

if 'gb_rev' in globals() and not gb_rev.empty:
    revenue_graph = px.bar(gb_rev,x='month_year',y='amount',color='sales_name',barmode='group')
    revenue_graph.show()
