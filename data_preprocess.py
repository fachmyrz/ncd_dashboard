import numpy as np
import pandas as pd
import geopy.distance
from sklearn.cluster import KMeans
from kneed import KneeLocator
import streamlit as st
from data_load import load_all_inputs

def _clean_df_visit(df_visit: pd.DataFrame) -> pd.DataFrame:
    df = df_visit.copy()
    df = df[['Employee Name','Client Name','Date Time Start','Date Time End','Note Start','Note End','Longitude Start','Latitude Start']]
    df.rename(columns={
        'Employee Name':'employee_name',
        'Client Name':'client_name',
        'Date Time Start':'date_time_start',
        'Date Time End':'date_time_end',
        'Note Start':'note_start',
        'Note End':'note_end',
        'Longitude Start':'long',
        'Latitude Start':'lat'
    }, inplace=True)
    df['time_start'] = df['date_time_start'].astype(str).apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
    df['time_end'] = df['date_time_end'].astype(str).apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
    df['date'] = df['date_time_start'].astype(str).apply(lambda x: x.split('@')[0] if '@' in x else np.nan)
    df['date'] = df['date'].astype(str).str.strip()
    df['date'] = pd.to_datetime(df['date'], format='%d %b %Y', errors='coerce').dt.date
    df.drop(columns=['date_time_start','date_time_end'], inplace=True)
    df['time_start'] = df['time_start'].astype(str).str.strip()
    df['time_end'] = df['time_end'].astype(str).str.strip()
    df['time_start'] = pd.to_datetime(df['time_start'], errors='coerce').dt.time
    df['time_end'] = pd.to_datetime(df['time_end'], errors='coerce').dt.time
    df['duration'] = (
        pd.to_datetime(df['time_end'].astype(str), errors='coerce') -
        pd.to_datetime(df['time_start'].astype(str), errors='coerce')
    ).dt.total_seconds() / 60
    return df

def _get_summary_data(df_visit_clean: pd.DataFrame, pick_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = df_visit_clean[df_visit_clean['date'] >= pd.to_datetime(pick_date).date()].copy()
    summary['lat'] = summary['lat'].astype(float)
    summary['long'] = summary['long'].astype(float)
    summary.reset_index(drop=True, inplace=True)
    rows = []
    for dt in summary['date'].dropna().unique():
        one_day = summary[summary['date'] == dt]
        for name in one_day['employee_name'].dropna().unique():
            temp = one_day[one_day.employee_name == name][['date','employee_name','lat','long','time_start','time_end']].reset_index(drop=True)
            if len(temp) > 1:
                dist = []
                tgap = []
                for i in range(len(temp)-1):
                    dist.append(round(geopy.distance.geodesic(
                        (temp.loc[i+1,'lat'], temp.loc[i+1,'long']),
                        (temp.loc[i,'lat'], temp.loc[i,'long'])
                    ).km, 2))
                    tgap.append(
                        (pd.to_datetime(str(temp.loc[i+1,'time_start'])) -
                         pd.to_datetime(str(temp.loc[i,'time_start']))).total_seconds() / 60
                    )
                avg_speed = round(sum(dist) / sum(tgap), 2) if sum(tgap) != 0 else 0
                rows.append([dt, name, len(temp), round(np.mean(dist),2), round(np.mean(tgap),2), avg_speed])
            else:
                rows.append([dt, name, len(temp), 0.0, 0.0, 0.0])
    cols = ['date','employee_name','ctd_visit','avg_distance_km','avg_time_between_minute','avg_speed_kmpm']
    data = pd.DataFrame(rows, columns=cols)
    data['month_year'] = data['date'].astype(str).str.slice(0,7)
    return summary, data

def _get_distance_dealer(kmeans, cluster_idx, lat, lng):
    return geopy.distance.geodesic((kmeans.cluster_centers_[cluster_idx,0],
                                    kmeans.cluster_centers_[cluster_idx,1]),
                                   (lat, lng)).km

@st.cache_data(show_spinner=True, ttl=900)
def prepare_data(pick_date: str = "2024-11-01"):
    inp = load_all_inputs()
    cluster_left = inp["cluster_left"]
    location_detail = inp["location_detail"]
    df_visit_raw = inp["df_visit"]
    df_dealer = inp["df_dealer"]
    running_order = inp["running_order"]
    df_visit = _clean_df_visit(df_visit_raw)
    summary, data_sum = _get_summary_data(df_visit, pick_date)
    filter_data = []
    for name in summary['employee_name'].dropna().unique():
        lat_long = summary[summary.employee_name == name][['lat','long']]
        if lat_long.empty:
            continue
        min_lat = lat_long['lat'].min()
        max_lat = lat_long['lat'].max()
        min_long = lat_long['long'].min()
        max_long = lat_long['long'].max()
        lat_km = geopy.distance.geodesic((max_lat, min_long), (min_lat, min_long)).km
        long_km = geopy.distance.geodesic((min_lat, max_long), (min_lat, min_long)).km
        area = lat_km * long_km
        filter_data.append([name, min_lat, max_lat, min_long, max_long, area])
    area_coverage = pd.DataFrame(
        data=filter_data,
        columns=['employee_name','min_lat','max_lat','min_long','max_long','area']
    ).astype({
        'min_lat':'float','max_lat':'float','min_long':'float','max_long':'float'
    })
    sum_data = []
    avail_data = []
    cluster_centers = []
    dealers = df_dealer.copy()
    dealers['business_type'] = "Car"
    dealers = dealers[['id_dealer_outlet','brand','business_type','city','name','state','latitude','longitude']]
    dealers = dealers[dealers.business_type.isin(['Car','Bike'])]
    dealers = dealers.dropna().reset_index(drop=True)
    dealers['latitude'] = dealers['latitude'].astype(str).str.replace(',', '', regex=False).astype(float)
    dealers['longitude'] = dealers['longitude'].astype(str).str.replace(',', '', regex=False).str.strip('.').astype(float)
    for name in area_coverage.employee_name.unique():
        bbox = area_coverage[area_coverage.employee_name == name]
        if bbox.empty:
            continue
        in_area = dealers[
            dealers.latitude.between(bbox.min_lat.values[0], bbox.max_lat.values[0]) &
            dealers.longitude.between(bbox.min_long.values[0], bbox.max_long.values[0])
        ]
        s = summary[summary.employee_name == name][['date','client_name','lat','long']].rename(
            columns={'lat':'latitude','long':'longitude'}
        )
        s['sales_name'] = name
        avail = in_area[['id_dealer_outlet','brand','business_type','city','name','latitude','longitude']].copy()
        avail['tag'] = 'avail'
        avail['sales_name'] = name
        kmeans = None
        if len(s) >= 2:
            wcss = []
            k_range = list(range(4, min(9, len(s))))
            if not k_range:
                k_range = [2]
            for k in k_range:
                km = KMeans(n_clusters=k, n_init="auto").fit(s[['latitude','longitude']].to_numpy())
                wcss.append(km.inertia_)
            knee = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
            n_cluster = knee.elbow if knee.elbow is not None else (k_range[0] if k_range else 2)
            kmeans = KMeans(n_clusters=n_cluster, n_init="auto").fit(s[['latitude','longitude']].to_numpy())
            s['cluster'] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters=1, n_init="auto").fit([[s['latitude'].mean() or 0, s['longitude'].mean() or 0]])
            s['cluster'] = 0
        for i in range(len(kmeans.cluster_centers_)):
            avail[f'dist_center_{i}'] = avail.apply(
                lambda x: _get_distance_dealer(kmeans, i, x.latitude, x.longitude), axis=1
            )
        clust_df = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude','longitude'])
        clust_df['sales_name'] = name
        clust_df['cluster'] = range(len(kmeans.cluster_centers_))
        cluster_centers.append(clust_df)
        avail_data.append(avail)
        sum_data.append(s)
    sum_df = pd.concat(sum_data, ignore_index=True) if sum_data else pd.DataFrame()
    avail_df = pd.concat(avail_data, ignore_index=True) if avail_data else pd.DataFrame()
    clust_df = pd.concat(cluster_centers, ignore_index=True) if cluster_centers else pd.DataFrame()
    active_order = running_order[['Dealer Id','Dealer Name','IsActive','End Date']].copy()
    active_order = active_order[active_order.IsActive == "1"]
    active_order['End Date'] = pd.to_datetime(active_order['End Date'], errors='coerce')
    active_order['Dealer Id'] = pd.to_numeric(active_order['Dealer Id'], errors='coerce').astype('Int64')
    active_order_group = (active_order
        .groupby(['Dealer Id','Dealer Name'])
        .agg({'End Date':'min'})
        .reset_index()
        .rename(columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','End Date':'nearest_end_date'})
    )
    run_order = running_order[['Dealer Id','Dealer Name','LMS Id','IsActive']].rename(
        columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','LMS Id':'joined_dse','IsActive':'active_dse'}
    )
    run_order['id_dealer_outlet'] = pd.to_numeric(run_order['id_dealer_outlet'], errors='coerce').astype('Int64')
    run_order['active_dse'] = pd.to_numeric(run_order['active_dse'], errors='coerce').astype('Int64')
    run_order = run_order[~run_order.id_dealer_outlet.isna()]
    grouped_run_order = (run_order
        .groupby(['id_dealer_outlet','dealer_name'])
        .agg({'joined_dse':'count','active_dse':'sum'})
        .reset_index()
    )
    run_order_group = pd.merge(grouped_run_order, active_order_group, how='left',
                               on=['id_dealer_outlet','dealer_name'])
    if not run_order_group.empty:
        run_order_group['id_dealer_outlet'] = run_order_group['id_dealer_outlet'].astype(int)
    if not avail_df.empty:
        avail_df['id_dealer_outlet'] = pd.to_numeric(avail_df['id_dealer_outlet'], errors='coerce').astype('Int64')
    if not avail_df.empty:
        numeric_cols = [c for c in avail_df.columns if str(c).startswith('dist_center_')]
        if numeric_cols:
            min_values = avail_df[numeric_cols].fillna(1e9).min(axis=1)
            avail_df[numeric_cols] = avail_df[numeric_cols].where(avail_df[numeric_cols].eq(min_values, axis=0), np.nan)
    avail_df_merge = pd.merge(avail_df, run_order_group.drop(columns=['dealer_name'], errors='ignore'),
                              how='left', on='id_dealer_outlet')
    avail_df_merge = pd.merge(
        avail_df_merge,
        location_detail[['City','Cluster']].rename(columns={'City':'city','Cluster':'cluster'}),
        how='left', on='city'
    )
    kpi_map = (cluster_left[cluster_left.get('Category','') == "Car"]
               .replace({'CHERY':'Chery','Kia':'KIA'}))
    kpi_map = kpi_map.rename(columns={
        'Cluster':'cluster',
        'Brand':'brand',
        'Daily_Gen':'daily_gen',
        'Daily_Need':'daily_need',
        'Delta':'delta',
        'Tag':'availability'
    })
    avail_df_merge = pd.merge(
        avail_df_merge,
        kpi_map[['cluster','brand','daily_gen','daily_need','delta','availability']],
        how='left', on=['brand','cluster']
    )
    if 'nearest_end_date' in avail_df_merge.columns:
        avail_df_merge['tag'] = np.where(avail_df_merge['nearest_end_date'].isna(), 'Not Active', 'Active')
    else:
        avail_df_merge['tag'] = 'Not Active'
    return {
        "summary": summary,
        "data_sum": data_sum,
        "area_coverage": area_coverage,
        "sum_df": sum_df,
        "avail_df": avail_df,
        "clust_df": clust_df,
        "avail_df_merge": avail_df_merge
    }
