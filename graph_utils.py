import plotly.express as px

def build_summary_figures(pick_date=None):
    try:
        from data_preprocess import get_summary_data
    except Exception:
        return None, None, None, None
    try:
        summary, data = get_summary_data(pick_date) if pick_date else get_summary_data()
    except Exception:
        return None, None, None, None
    if data is None or data.empty:
        return None, None, None, None
    gb_sum = data.groupby(["month_year", "employee_name"], as_index=False).agg({"ctd_visit":"sum","avg_distance_km":"mean","avg_time_between_minute":"mean","avg_speed_kmpm":"median"})
    try:
        ctd_visit = px.line(gb_sum, x="month_year", y="ctd_visit", color="employee_name", markers=True)
        avg_distance = px.line(gb_sum, x="month_year", y="avg_distance_km", color="employee_name", markers=True)
        avg_walk_time = px.line(gb_sum, x="month_year", y="avg_time_between_minute", color="employee_name", markers=True)
        avg_speed = px.line(gb_sum, x="month_year", y="avg_speed_kmpm", color="employee_name", markers=True)
    except Exception:
        return None, None, None, None
    ctd_visit.update_layout(title_text="Total Visit per Bulan")
    avg_distance.update_layout(title_text="Rata-rata Jarak per Visit")
    avg_walk_time.update_layout(title_text="Rata-rata Waktu antar Visit")
    avg_speed.update_layout(title_text="Rata-rata Kecepatan antar Visit")
    return ctd_visit, avg_distance, avg_walk_time, avg_speed
