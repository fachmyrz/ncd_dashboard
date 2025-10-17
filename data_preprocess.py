import pandas as pd
import numpy as np
import geopy.distance
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from kneed import KneeLocator
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def enhanced_preprocessing(_dealers, _visits, _running_order, _location_detail, _cluster_left):
    """Enhanced preprocessing with clustering and performance metrics"""
    
    # Clean dealer data
    dealers_clean = _dealers.copy()
    dealers_clean['business_type'] = "Car"
    dealers_clean = dealers_clean[['id_dealer_outlet','brand','business_type','city','name','state','latitude','longitude']]
    dealers_clean = dealers_clean[dealers_clean.business_type.isin(['Car','Bike'])]
    dealers_clean = dealers_clean.dropna().reset_index(drop=True)
    
    # Convert coordinates
    dealers_clean['latitude'] = pd.to_numeric(dealers_clean['latitude'].astype(str).str.replace(',', ''), errors='coerce')
    dealers_clean['longitude'] = pd.to_numeric(dealers_clean['longitude'].astype(str).str.replace(',', '').str.strip('.'), errors='coerce')
    dealers_clean = dealers_clean.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

    # Clean visit data
    visits_clean = _visits.copy()
    visits_clean = visits_clean[['Employee Name','Client Name','Date Time Start','Date Time End',
                               'Note Start','Note End','Longitude Start','Latitude Start']]
    
    visits_clean.rename(columns={
        'Employee Name':'employee_name',
        'Client Name':'client_name',
        'Date Time Start':'date_time_start',
        'Date Time End':'date_time_end',
        'Note Start':'note_start',
        'Note End':'note_end',
        'Longitude Start':'long',
        'Latitude Start':'lat'
    }, inplace=True)
    
    # Parse datetime
    visits_clean['time_start'] = visits_clean['date_time_start'].astype(str).apply(
        lambda x: x.split('@')[1] if '@' in x else np.nan)
    visits_clean['time_end'] = visits_clean['date_time_end'].astype(str).apply(
        lambda x: x.split('@')[1] if '@' in x else np.nan)
    visits_clean['date'] = visits_clean['date_time_start'].astype(str).apply(
        lambda x: x.split('@')[0] if '@' in x else np.nan)
    
    visits_clean['date'] = pd.to_datetime(visits_clean['date'].str.strip(), format='%d %b %Y', errors='coerce').dt.date
    visits_clean = visits_clean.dropna(subset=['date']).reset_index(drop=True)
    
    # Calculate duration
    visits_clean['duration_minutes'] = (
        pd.to_datetime(visits_clean['time_end'].astype(str)) - 
        pd.to_datetime(visits_clean['time_start'].astype(str))
    ).dt.total_seconds() / 60
    
    # Performance metrics calculation
    def calculate_performance_metrics(visits_df):
        metrics = []
        for (date, employee), group in visits_df.groupby(['date', 'employee_name']):
            if len(group) > 1:
                distances = []
                times_between = []
                group = group.sort_values('time_start').reset_index(drop=True)
                
                for i in range(len(group)-1):
                    try:
                        dist = geopy.distance.geodesic(
                            (group.loc[i, 'lat'], group.loc[i, 'long']),
                            (group.loc[i+1, 'lat'], group.loc[i+1, 'long'])
                        ).km
                        distances.append(dist)
                        
                        time_diff = (
                            pd.to_datetime(str(group.loc[i+1, 'time_start'])) - 
                            pd.to_datetime(str(group.loc[i, 'time_start']))
                        ).total_seconds() / 60
                        times_between.append(time_diff)
                    except:
                        continue
                
                avg_distance = np.mean(distances) if distances else 0
                avg_time_between = np.mean(times_between) if times_between else 0
                avg_speed = avg_distance / avg_time_between if avg_time_between > 0 else 0
                
                metrics.append({
                    'date': date,
                    'employee_name': employee,
                    'total_visits': len(group),
                    'avg_distance_km': round(avg_distance, 2),
                    'avg_time_between_min': round(avg_time_between, 2),
                    'avg_speed_km_per_min': round(avg_speed, 2),
                    'total_duration_min': group['duration_minutes'].sum()
                })
        
        return pd.DataFrame(metrics)
    
    performance_df = calculate_performance_metrics(visits_clean)
    
    # Clustering for dealer recommendations
    def create_dealer_clusters(dealers_df, visits_df):
        clustered_data = []
        
        for employee in visits_df['employee_name'].unique():
            employee_visits = visits_df[visits_df['employee_name'] == employee]
            
            if len(employee_visits) < 2:
                continue
                
            # Use elbow method to find optimal clusters
            coords = employee_visits[['lat', 'long']].dropna().values
            if len(coords) < 4:
                continue
                
            wcss = []
            for k in range(2, min(8, len(coords))):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(coords)
                wcss.append(kmeans.inertia_)
            
            if len(wcss) > 1:
                kneedle = KneeLocator(range(2, min(8, len(coords))), wcss, curve='convex', direction='decreasing')
                n_clusters = kneedle.elbow or 3
            else:
                n_clusters = 2
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            employee_visits['cluster'] = kmeans.fit_predict(coords)
            
            # Assign dealers to clusters
            for cluster_id in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_dealers = dealers_df.copy()
                
                cluster_dealers['distance_to_center'] = cluster_dealers.apply(
                    lambda row: geopy.distance.geodesic(
                        (row['latitude'], row['longitude']),
                        (cluster_center[0], cluster_center[1])
                    ).km, axis=1
                )
                
                cluster_dealers['employee_name'] = employee
                cluster_dealers['cluster_id'] = cluster_id
                clustered_data.append(cluster_dealers)
        
        return pd.concat(clustered_data, ignore_index=True) if clustered_data else pd.DataFrame()

    dealer_clusters = create_dealer_clusters(dealers_clean, visits_clean)
    
    return {
        'dealers_clean': dealers_clean,
        'visits_clean': visits_clean,
        'performance_df': performance_df,
        'dealer_clusters': dealer_clusters
    }
