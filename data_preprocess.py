import pandas as pd
import numpy as np
import geopy.distance
from data_load import *
from sklearn.cluster import KMeans
from datetime import datetime, timedelta

def normalize_cols(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(' ', '_').str.lower()
    return df

df_dealer = normalize_cols(df_dealer) if 'df_dealer' in globals() else pd.DataFrame()
expected_dealer_cols = ['id_dealer_outlet','brand','business_type','city','name','state','latitude','longitude']
for c in expected_dealer_cols:
    if c not in df_dealer.columns:
        df_dealer[c] = np.nan
df_dealer = df_dealer[expected_dealer_cols].copy()
df_dealer['business_type'] = df_dealer['business_type'].fillna('Car')
df_dealer = df_dealer[df_dealer.business_type.isin(['Car','Bike'])]
df_dealer = df_dealer.dropna(subset=['latitude','longitude'], how='all').reset_index(drop=True)
if not df_dealer.empty:
    df_dealer['latitude'] = df_dealer['latitude'].astype(str).str.replace(',', '', regex=False).str.strip()
    df_dealer['longitude'] = df_dealer['longitude'].astype(str).str.replace(',', '', regex=False).str.strip('.').str.strip()
    df_dealer['latitude'] = pd.to_numeric(df_dealer['latitude'], errors='coerce')
    df_dealer['longitude'] = pd.to_numeric(df_dealer['longitude'], errors='coerce')

df_visit = normalize_cols(df_visit) if 'df_visit' in globals() else pd.DataFrame()
expected_visit_cols = ['employee_name','client_name','date_time_start','date_time_end','note_start','note_end','longitude_start','latitude_start']
for c in expected_visit_cols:
    if c not in df_visit.columns:
        df_visit[c] = np.nan
df_visit = df_visit[expected_visit_cols].copy()
df_visit.rename(columns={'longitude_start':'long','latitude_start':'lat'}, inplace=True)

def parse_datetime_field(x):
    if pd.isna(x):
        return (pd.NaT, pd.NaT)
    s = str(x).strip()
    if '@' in s:
        parts = s.split('@', 1)
        date_str = parts[0].strip()
        time_str = parts[1].strip()
        dt_date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        if pd.isna(dt_date):
            dt_date = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
        tmp = pd.to_datetime(time_str, errors='coerce')
        dt_time = tmp.time() if not pd.isna(tmp) else pd.NaT
        return (dt_date.date() if not pd.isna(dt_date) else pd.NaT, dt_time)
    dt = pd.to_datetime(s, dayfirst=False, errors='coerce')
    if pd.isna(dt):
        dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
    if not pd.isna(dt):
        return (dt.date(), dt.time())
    parts = s.split()
    if len(parts) >= 2:
        date_part = ' '.join(parts[:-1]).strip()
        time_part = parts[-1].strip()
        dt_date = pd.to_datetime(date_part, dayfirst=True, errors='coerce')
        if pd.isna(dt_date):
            dt_date = pd.to_datetime(date_part, dayfirst=False, errors='coerce')
        tmp = pd.to_datetime(time_part, errors='coerce')
        dt_time = tmp.time() if not pd.isna(tmp) else pd.NaT
        if not pd.isna(dt_date):
            return (dt_date.date(), dt_time)
    return (pd.NaT, pd.NaT)

parsed_start = df_visit['date_time_start'].apply(parse_datetime_field)
parsed_end = df_visit['date_time_end'].apply(parse_datetime_field)
if not df_visit.empty:
    df_visit[['date','time_start']] = pd.DataFrame(parsed_start.tolist(), index=df_visit.index)
    df_visit[['date_end','time_end']] = pd.DataFrame(parsed_end.tolist(), index=df_visit.index)
else:
    df_visit['date'] = pd.NaT
    df_visit['time_start'] = pd.NaT
    df_visit['time_end'] = pd.NaT
df_visit['date'] = pd.to_datetime(df_visit['date'], errors='coerce').dt.date
df_visit['time_start'] = df_visit['time_start'].where(df_visit['time_start'].notna(), pd.NaT)
df_visit['time_end'] = df_visit['time_end'].where(df_visit['time_end'].notna(), pd.NaT)
df_visit = df_visit.drop(columns=['date_time_start','date_time_end','date_end'], errors='ignore')
df_visit['duration'] = (pd.to_datetime(df_visit['time_end'].astype(str), errors='coerce') - pd.to_datetime(df_visit['time_start'].astype(str), errors='coerce')).dt.total_seconds() / 60
if 'lat' in df_visit.columns:
    df_visit['lat'] = pd.to_numeric(df_visit['lat'], errors='coerce')
else:
    df_visit['lat'] = pd.Series(dtype='float')
if 'long' in df_visit.columns:
    df_visit['long'] = pd.to_numeric(df_visit['long'], errors='coerce')
else:
    df_visit['long'] = pd.Series(dtype='float')

def get_summary_data(pick_date=None):
    if pick_date is None:
        if 'df_visit' in globals() and not df_visit.empty and 'date' in df_visit.columns:
            try:
                min_date = pd.to_datetime(df_visit['date'], errors='coerce').min()
                if pd.isna(min_date):
                    pick_date = "1970-01-01"
                else:
                    pick_date = min_date.strftime("%Y-%m-%d")
            except Exception:
                pick_date = "1970-01-01"
        else:
            pick_date = "1970-01-01"
    summary = df_visit[df_visit['date'] >= pd.to_datetime(pick_date).date()].copy() if not df_visit.empty else pd.DataFrame()
    if not summary.empty:
        summary['lat'] = pd.to_numeric(summary['lat'], errors='coerce')
        summary['long'] = pd.to_numeric(summary['long'], errors='coerce')
        summary.reset_index(drop=True,inplace=True)
    data = []
    if not summary.empty:
        for dates in summary['date'].unique():
            for name in summary['employee_name'].unique():
                temp = summary[(summary.employee_name == name)&(summary['date'] == dates)].reset_index(drop=True)
                temp = temp[['date','employee_name','lat','long','time_start','time_end']].copy()
                if len(temp) > 1:
                    dist = []
                    time_between = []
                    for i in range(len(temp)-1):
                        try:
                            d = round(geopy.distance.geodesic((temp.loc[i+1,'lat'],temp.loc[i+1,'long']), (temp.loc[i,'lat'],temp.loc[i,'long'])).km,2)
                        except Exception:
                            d = 0
                        dist.append(d)
                        try:
                            tb = (pd.to_datetime(str(temp.loc[i+1,'time_start']), errors='coerce') - pd.to_datetime(str(temp.loc[i,'time_start']), errors='coerce')).total_seconds()/60
                        except Exception:
                            tb = 0
                        time_between.append(tb)
                    avg_speed = round(sum(dist) / sum(time_between),2) if sum(time_between) != 0 else 0
                    data.append([dates,name,len(temp),round(np.mean(dist),2),round(np.mean(time_between),2), avg_speed])
                else:
                    data.append([dates,name,len(temp),0,0,0])
    cols = ['date','employee_name','ctd_visit','avg_distance_km','avg_time_between_minute','avg_speed_kmpm']
    data = pd.DataFrame(data,columns=cols) if len(data)>0 else pd.DataFrame(columns=cols)
    if not data.empty:
        data['month_year'] = data['date'].astype(str).apply(lambda x: x.split('-')[0]+'-'+x.split('-')[1])
    return summary,data

summary,data_sum = get_summary_data()

filter_data = []
if not summary.empty:
    for name in summary.employee_name.unique():
        lat_long = summary[summary.employee_name == name][['lat','long']].copy()
        min_lat = lat_long['lat'].min()
        max_lat = lat_long['lat'].max()
        min_long = lat_long['long'].min()
        max_long = lat_long['long'].max()
        lat_ = geopy.distance.geodesic((max_lat,min_long),(min_lat,min_long)).km if pd.notna(max_lat) and pd.notna(min_lat) and pd.notna(min_long) else 0
        long_ = geopy.distance.geodesic((min_lat,max_long),(min_lat,min_long)).km if pd.notna(min_lat) and pd.notna(max_long) and pd.notna(min_long) else 0
        area = lat_ * long_
        filter_data.append([name,min_lat,max_lat,min_long,max_long,area])
area_coverage = pd.DataFrame(data=filter_data,columns=['employee_name','min_lat','max_lat','min_long','max_long','area']) if len(filter_data)>0 else pd.DataFrame(columns=['employee_name','min_lat','max_lat','min_long','max_long','area'])
for c in ['min_lat','min_long','max_lat','max_long']:
    if c in area_coverage.columns:
        area_coverage[c] = pd.to_numeric(area_coverage[c], errors='coerce')

sum_data = []
avail_data = []
cluster_center = []

def compute_distance(centers, cluster_index, lat, long):
    try:
        return geopy.distance.geodesic((centers[cluster_index][0], centers[cluster_index][1]), (lat,long)).km
    except Exception:
        return np.nan

for name in area_coverage.employee_name.unique():
    data_ = area_coverage[area_coverage.employee_name == name]
    get_dealer = df_dealer[(df_dealer.latitude.between(data_.min_lat.values[0],data_.max_lat.values[0])) & (df_dealer.longitude.between(data_.min_long.values[0],data_.max_long.values[0]))].copy() if not df_dealer.empty else pd.DataFrame()
    sum_ = summary[summary.employee_name == name][['date','client_name','lat','long']].copy() if not summary.empty else pd.DataFrame()
    sum_.rename(columns={'lat':'latitude','long':'longitude'},inplace=True)
    sum_['sales_name'] = name
    avail_ = get_dealer[['id_dealer_outlet','brand','business_type','city','name','latitude','longitude']].copy() if not get_dealer.empty else pd.DataFrame(columns=['id_dealer_outlet','brand','business_type','city','name','latitude','longitude'])
    avail_['tag'] = 'avail'
    avail_['sales_name'] = name
    kmeans_obj = None
    if len(sum_) >= 1:
        if len(sum_) >= 4:
            wcss = []
            max_k = min(8, len(sum_))
            for i in range(2, max_k+1):
                X = list(zip(sum_['latitude'],sum_['longitude']))
                try:
                    k_temp = KMeans(n_clusters=i, random_state=42).fit(X)
                    wcss.append(k_temp.inertia_)
                except Exception:
                    wcss.append(None)
            valid_wcss = [v for v in wcss if v is not None]
            if len(valid_wcss) >= 1:
                try:
                    n_cluster = min(4, len(sum_))
                except Exception:
                    n_cluster = min(4, len(sum_))
            else:
                n_cluster = min(4, len(sum_))
        else:
            n_cluster = max(1, len(sum_))
        n_cluster = max(1, min(n_cluster, len(sum_)))
        try:
            data_coords = list(zip(sum_['latitude'],sum_['longitude']))
            kmeans_obj = KMeans(n_clusters=n_cluster, random_state=42).fit(data_coords)
            sum_['cluster'] = kmeans_obj.labels_
            centers = kmeans_obj.cluster_centers_
            for i in range(len(centers)):
                avail_[f'dist_center_{i}'] = avail_.apply(lambda x, ci=i, centers=centers: compute_distance(centers, ci, x.latitude, x.longitude), axis=1)
        except Exception:
            sum_['cluster'] = 0
    else:
        sum_['cluster'] = 0
    if kmeans_obj is not None:
        clust_ = pd.DataFrame(kmeans_obj.cluster_centers_,columns=['latitude','longitude'])
        clust_['sales_name'] = name
        clust_['cluster'] = range(len(kmeans_obj.cluster_centers_))
    else:
        if not sum_.empty and 'latitude' in sum_ and 'longitude' in sum_:
            clust_ = pd.DataFrame([[sum_['latitude'].astype(float).mean(), sum_['longitude'].astype(float).mean()]], columns=['latitude','longitude'])
            clust_['sales_name'] = name
            clust_['cluster'] = [0]
        else:
            clust_ = pd.DataFrame([[0.0,0.0]], columns=['latitude','longitude'])
            clust_['sales_name'] = name
            clust_['cluster'] = [0]
    cluster_center.append(clust_)
    avail_data.append(avail_)
    sum_data.append(sum_)

sum_df = pd.concat(sum_data, ignore_index=True) if len(sum_data)>0 else pd.DataFrame(columns=['date','client_name','latitude','longitude','sales_name','cluster'])
avail_df = pd.concat(avail_data, ignore_index=True) if len(avail_data)>0 else pd.DataFrame()
clust_df = pd.concat(cluster_center, ignore_index=True) if len(cluster_center)>0 else pd.DataFrame(columns=['latitude','longitude','sales_name','cluster'])

running_order = normalize_cols(running_order) if 'running_order' in globals() else pd.DataFrame()
running_expected = ['dealer_id','dealer_name','isactive','end_date','lms_id']
for c in running_expected:
    if c not in running_order.columns:
        running_order[c] = np.nan

active_order = running_order[['dealer_id','dealer_name','isactive','end_date']].copy() if not running_order.empty else pd.DataFrame()
active_order = active_order[active_order.isactive == "1"] if not active_order.empty else active_order
if not active_order.empty:
    active_order['end_date'] = pd.to_datetime(active_order['end_date'], errors='coerce')
    active_order['dealer_id'] = pd.to_numeric(active_order['dealer_id'], errors='coerce')
    active_order_group = active_order.groupby(['dealer_id','dealer_name']).agg({'end_date':'min'}).reset_index()
    active_order_group.rename(columns={'dealer_id':'id_dealer_outlet','dealer_name':'dealer_name','end_date':'nearest_end_date'},inplace=True)
else:
    active_order_group = pd.DataFrame(columns=['id_dealer_outlet','dealer_name','nearest_end_date'])

run_order = running_order[['dealer_id','dealer_name','lms_id','isactive']].copy() if not running_order.empty else pd.DataFrame()
run_order.rename(columns={'dealer_id':'id_dealer_outlet','dealer_name':'dealer_name','lms_id':'joined_dse','isactive':'active_dse'},inplace=True)
if not run_order.empty:
    run_order['id_dealer_outlet'] = pd.to_numeric(run_order['id_dealer_outlet'], errors='coerce').astype('Int64')
    run_order['active_dse'] = pd.to_numeric(run_order['active_dse'], errors='coerce').astype('Int64')
    run_order = run_order[~run_order.id_dealer_outlet.isna()]
grouped_run_order = run_order.groupby(['id_dealer_outlet','dealer_name']).agg({'joined_dse':'count','active_dse':'sum'}).reset_index() if not run_order.empty else pd.DataFrame(columns=['id_dealer_outlet','dealer_name','joined_dse','active_dse'])
run_order_group = pd.merge(grouped_run_order,active_order_group,how='left',on=['id_dealer_outlet','dealer_name']) if not grouped_run_order.empty else pd.DataFrame(columns=['id_dealer_outlet','dealer_name','joined_dse','active_dse','nearest_end_date'])
if not run_order_group.empty:
    run_order_group['id_dealer_outlet'] = run_order_group['id_dealer_outlet'].astype(int)
if not avail_df.empty and 'id_dealer_outlet' in avail_df.columns:
    avail_df['id_dealer_outlet'] = pd.to_numeric(avail_df['id_dealer_outlet'], errors='coerce').fillna(0).astype(int)

if not avail_df.empty and avail_df.shape[1] > 9:
    min_values = avail_df.fillna(100000000).iloc[:, 9:].min(axis=1)
    avail_df.iloc[:, 9:] = avail_df.iloc[:, 9:].where(avail_df.iloc[:, 9:].eq(min_values, axis=0), np.nan)

if not avail_df.empty and not run_order_group.empty:
    avail_df_merge = pd.merge(avail_df, run_order_group.drop(columns=['dealer_name']), how='left', on='id_dealer_outlet')
else:
    avail_df_merge = avail_df.copy() if not avail_df.empty else pd.DataFrame()

location_detail = normalize_cols(location_detail) if 'location_detail' in globals() else pd.DataFrame()
if not location_detail.empty and len(location_detail.columns) >= 2:
    location_detail = location_detail.rename(columns={location_detail.columns[0]:'city', location_detail.columns[1]:'cluster'})
if not avail_df_merge.empty and not location_detail.empty:
    avail_df_merge = pd.merge(avail_df_merge, location_detail[['city','cluster']], how='left', on='city')
else:
    if 'city' not in avail_df_merge.columns:
        avail_df_merge['city'] = np.nan
    if 'cluster' not in avail_df_merge.columns:
        avail_df_merge['cluster'] = np.nan

cluster_left = normalize_cols(cluster_left) if 'cluster_left' in globals() else pd.DataFrame()
if not cluster_left.empty:
    cluster_left = cluster_left.replace({'chery':'Chery','kia':'KIA'})
    cluster_left = cluster_left.rename(columns={cluster_left.columns[0]:'category'}).rename(columns={'cluster':'cluster','brand':'brand','daily_gen':'daily_gen','daily_need':'daily_need','delta':'delta','tag':'availability'})
if not avail_df_merge.empty and not cluster_left.empty:
    avail_df_merge = pd.merge(avail_df_merge, cluster_left[['cluster','brand','daily_gen','daily_need','delta','availability']], how='left', on=['brand','cluster'])
else:
    if 'availability' not in avail_df_merge.columns:
        avail_df_merge['availability'] = np.nan

if 'avail_df_merge' not in globals() or avail_df_merge is None:
    avail_df_merge = pd.DataFrame()
if 'nearest_end_date' not in avail_df_merge.columns:
    avail_df_merge['nearest_end_date'] = pd.NA
if 'tag' not in avail_df_merge.columns:
    avail_df_merge['tag'] = pd.NA
avail_df_merge['tag'] = np.where(pd.isna(avail_df_merge['nearest_end_date']), 'Not Active', avail_df_merge['tag'])

sales_data = normalize_cols(sales_data) if 'sales_data' in globals() else pd.DataFrame()
if not sales_data.empty:
    sd_expected = ['date_of_sales_based_on_image_proof','sales_id','amount']
    for c in sd_expected:
        if c not in sales_data.columns:
            sales_data[c] = np.nan
    rev_data = sales_data[['date_of_sales_based_on_image_proof','sales_id','amount']].copy()
    rev_data.rename(columns={'date_of_sales_based_on_image_proof':'date','sales_id':'sales_name','amount':'amount'},inplace=True)
    rev_data['date'] = pd.to_datetime(rev_data['date'], errors='coerce')
    rev_data['amount'] = rev_data['amount'].astype(str).str.replace(',', '', regex=False)
    rev_data['amount'] = pd.to_numeric(rev_data['amount'], errors='coerce').fillna(0).astype(int)
    yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
    rev_data = rev_data[rev_data.date <= yesterday]
    gb_rev = rev_data[rev_data.date >= "2024-11-01"].copy()
    if not gb_rev.empty:
        gb_rev['month_year'] = gb_rev['date'].dt.to_period('M').astype(str)
        gb_rev = gb_rev.groupby(['month_year','sales_name']).agg({'amount':'sum'}).reset_index()
    else:
        gb_rev = pd.DataFrame(columns=['month_year','sales_name','amount'])
else:
    gb_rev = pd.DataFrame(columns=['month_year','sales_name','amount'])
