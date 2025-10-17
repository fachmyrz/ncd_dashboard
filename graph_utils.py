import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_performance_charts(performance_df, visits_df):
    """Create comprehensive performance charts"""
    
    charts = {}
    
    # Visits over time by employee
    if not visits_df.empty and 'employee_name' in visits_df.columns:
        visits_by_employee = visits_df.groupby(['date', 'employee_name']).size().reset_index(name='visits')
        fig_visits = px.line(visits_by_employee, x='date', y='visits', color='employee_name',
                           title='Daily Visits by Employee', markers=True)
        charts['visits_trend'] = fig_visits
    
    # Performance metrics
    if not performance_df.empty:
        # Average distance vs visits
        fig_scatter = px.scatter(performance_df, x='total_visits', y='avg_distance_km',
                               color='employee_name', size='total_visits',
                               title='Efficiency Analysis: Visits vs Distance',
                               hover_data=['avg_time_between_min'])
        charts['efficiency_scatter'] = fig_scatter
        
        # Performance comparison
        avg_metrics = performance_df.groupby('employee_name').agg({
            'total_visits': 'sum',
            'avg_distance_km': 'mean',
            'avg_speed_km_per_min': 'mean'
        }).reset_index()
        
        fig_radar = go.Figure()
        
        for _, employee in avg_metrics.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[employee['total_visits'], employee['avg_distance_km'], employee['avg_speed_km_per_min']],
                theta=['Total Visits', 'Avg Distance (km)', 'Avg Speed (km/min)'],
                fill='toself',
                name=employee['employee_name']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(avg_metrics['total_visits'].max(), 
                                                          avg_metrics['avg_distance_km'].max(),
                                                          avg_metrics['avg_speed_km_per_min'].max())])
            ),
            showlegend=True,
            title='Employee Performance Comparison'
        )
        charts['performance_radar'] = fig_radar
    
    return charts

def create_dealer_analysis_charts(dealers_df):
    """Create dealer analysis charts"""
    
    charts = {}
    
    if dealers_df.empty:
        return charts
    
    # Brand distribution
    brand_dist = dealers_df['brand'].value_counts().reset_index()
    brand_dist.columns = ['Brand', 'Count']
    fig_brand = px.pie(brand_dist, values='Count', names='Brand', 
                      title='Dealer Distribution by Brand')
    charts['brand_distribution'] = fig_brand
    
    # Geographic distribution
    city_dist = dealers_df['city'].value_counts().head(15).reset_index()
    city_dist.columns = ['City', 'Count']
    fig_city = px.bar(city_dist, x='City', y='Count', 
                     title='Top 15 Cities by Dealer Count')
    fig_city.update_layout(xaxis_tickangle=-45)
    charts['city_distribution'] = fig_city
    
    return charts
