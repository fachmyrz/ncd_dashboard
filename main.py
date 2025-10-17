import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from data_load import get_sheets
from data_preprocess import enhanced_preprocessing

# Page configuration
st.set_page_config(
    page_title="Dealer Penetration Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Dealer Penetration Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            sheets = get_sheets()
            processed_data = enhanced_preprocessing(
                sheets.get('df_dealer', pd.DataFrame()),
                sheets.get('df_visit', pd.DataFrame()),
                sheets.get('running_order', pd.DataFrame()),
                sheets.get('location_detail', pd.DataFrame()),
                sheets.get('cluster_left', pd.DataFrame())
            )
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    # Sidebar filters
    st.sidebar.markdown("## 🔧 Filters")
    
    # Employee filter
    employees = processed_data['visits_clean']['employee_name'].unique() if not processed_data['visits_clean'].empty else []
    selected_employee = st.sidebar.selectbox("Select Employee", options=['All'] + list(employees))
    
    # Date range filter
    if not processed_data['visits_clean'].empty:
        min_date = processed_data['visits_clean']['date'].min()
        max_date = processed_data['visits_clean']['date'].max()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Brand filter
    brands = processed_data['dealers_clean']['brand'].unique() if not processed_data['dealers_clean'].empty else []
    selected_brands = st.sidebar.multiselect("Brands", options=brands, default=brands[:3] if len(brands) > 3 else brands)
    
    # City filter
    cities = processed_data['dealers_clean']['city'].unique() if not processed_data['dealers_clean'].empty else []
    selected_cities = st.sidebar.multiselect("Cities", options=cities, default=cities[:3] if len(cities) > 3 else cities)

    # Main dashboard
    if processed_data['visits_clean'].empty or processed_data['dealers_clean'].empty:
        st.warning("No data available. Please check your data sources.")
        return

    # Key Metrics
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_visits = len(processed_data['visits_clean'])
        st.metric("Total Visits", f"{total_visits:,}")
    
    with col2:
        total_dealers = len(processed_data['dealers_clean'])
        st.metric("Total Dealers", f"{total_dealers:,}")
    
    with col3:
        active_employees = processed_data['visits_clean']['employee_name'].nunique()
        st.metric("Active Employees", active_employees)
    
    with col4:
        avg_visits_per_employee = total_visits / active_employees if active_employees > 0 else 0
        st.metric("Avg Visits/Employee", f"{avg_visits_per_employee:.1f}")

    # Performance Charts
    st.markdown('<h2 class="section-header">📈 Performance Analytics</h2>', unsafe_allow_html=True)
    
    if not processed_data['performance_df'].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Visits over time
            visits_trend = processed_data['visits_clean'].groupby('date').size().reset_index(name='visits')
            fig_visits = px.line(visits_trend, x='date', y='visits', title='Daily Visits Trend')
            st.plotly_chart(fig_visits, use_container_width=True)
        
        with col2:
            # Employee performance
            employee_perf = processed_data['performance_df'].groupby('employee_name').agg({
                'total_visits': 'sum',
                'avg_distance_km': 'mean'
            }).reset_index()
            
            fig_employee = px.bar(employee_perf, x='employee_name', y='total_visits', 
                                title='Total Visits by Employee')
            st.plotly_chart(fig_employee, use_container_width=True)

    # Dealer Map
    st.markdown('<h2 class="section-header">🗺️ Dealer Locations</h2>', unsafe_allow_html=True)
    
    # Filter dealers
    filtered_dealers = processed_data['dealers_clean'].copy()
    if selected_brands:
        filtered_dealers = filtered_dealers[filtered_dealers['brand'].isin(selected_brands)]
    if selected_cities:
        filtered_dealers = filtered_dealers[filtered_dealers['city'].isin(selected_cities)]
    
    if not filtered_dealers.empty:
        # Create map
        view_state = pdk.ViewState(
            latitude=filtered_dealers['latitude'].mean(),
            longitude=filtered_dealers['longitude'].mean(),
            zoom=10,
            pitch=50
        )
        
        dealer_layer = pdk.Layer(
            'ScatterplotLayer',
            data=filtered_dealers,
            get_position=['longitude', 'latitude'],
            get_color=[255, 0, 0, 160],
            get_radius=500,
            pickable=True
        )
        
        tooltip = {
            "html": "<b>Dealer:</b> {name}<br><b>Brand:</b> {brand}<br><b>City:</b> {city}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[dealer_layer],
            tooltip=tooltip
        ))

    # Dealer Analysis
    st.markdown('<h2 class="section-header">🏢 Dealer Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand distribution
        brand_dist = filtered_dealers['brand'].value_counts().reset_index()
        brand_dist.columns = ['Brand', 'Count']
        fig_brand = px.pie(brand_dist, values='Count', names='Brand', title='Dealer Distribution by Brand')
        st.plotly_chart(fig_brand, use_container_width=True)
    
    with col2:
        # City distribution
        city_dist = filtered_dealers['city'].value_counts().head(10).reset_index()
        city_dist.columns = ['City', 'Count']
        fig_city = px.bar(city_dist, x='City', y='Count', title='Top 10 Cities by Dealer Count')
        st.plotly_chart(fig_city, use_container_width=True)

    # Raw Data Section
    st.markdown('<h2 class="section-header">📋 Data Overview</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Dealers", "Visits", "Performance"])
    
    with tab1:
        st.dataframe(filtered_dealers, use_container_width=True)
    
    with tab2:
        st.dataframe(processed_data['visits_clean'], use_container_width=True)
    
    with tab3:
        if not processed_data['performance_df'].empty:
            st.dataframe(processed_data['performance_df'], use_container_width=True)
        else:
            st.info("No performance data available")

    # Recommendations Section
    st.markdown('<h2 class="section-header">💡 Recommendations</h2>', unsafe_allow_html=True)
    
    if not processed_data['dealer_clusters'].empty:
        # Show cluster-based recommendations
        cluster_recommendations = processed_data['dealer_clusters'].groupby(['employee_name', 'cluster_id']).agg({
            'name': 'count',
            'distance_to_center': 'mean'
        }).reset_index()
        
        st.write("**Dealer Clusters for Optimization:**")
        st.dataframe(cluster_recommendations, use_container_width=True)
    else:
        st.info("Cluster analysis data not available")

if __name__ == "__main__":
    main()
