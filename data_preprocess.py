import pandas as pd
import numpy as np
import geopy.distance
from data_load import *
from sklearn.cluster import KMeans
from kneed import KneeLocator
from datetime import datetime, timedelta

df_dealer = df_dealer.copy() if not df_dealer.empty else pd.DataFrame(columns=['id_dealer_outlet','brand','business_type','city','name','state','latitude','longitude'])
df_dealer['business_type'] = df_dealer.get('business_type', pd.Series(["Car"]*len(df_dealer)))
df_dealer = df_dealer[['id_dealer_outlet','brand','business_type','city','name','state','latitude','longitude']]
df_dealer = df_dealer[df_dealer.business_type.isin(['Car','Bike'])]
df_dealer = df_dealer.dropna().reset_index(drop=True)
if not df_dealer.empty:
    df_dealer['latitude'] = df_dealer['latitude'].str.replace(',', '', regex=False).astype(float)
    df_dealer['longitude'] = df_dealer['longitude'].str.replace(',', '', regex=False).str.strip('.').astype(float)

df_visit = df_visit.copy() if not df_visit.empty else pd.DataFrame(columns=['Employee Name','Client Name','Date Time Start','Date Time End','Note Start','Note End','Longitude Start','Latitude Start'])
df_visit = df_visit[['Employee Name','Client Name','Date Time Start','Date Time End','Note Start','Note End','Longitude Start','Latitude Start']]
df_visit.rename(columns={'Employee Name':'employee_name','Client Name':'client_name','Date Time Start':'date_time_start','Date Time End':'date_time_end','Note Start':'note_start','Note End':'note_end','Longitude Start':'long','Latitude Start':'lat'},inplace=True)
df_visit['time_start'] = df_visit['date_time_start'].astype(str).apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
df_visit['time_end'] = df_visit['date_time_end'].astype(str).apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
df_visit['date'] = df_visit['date_time_start'].astype(str).apply(lambda x: x.split('@')[0] if '@' in x else np.nan)
df_visit['date'] = df_visit['date'].str.strip()
df_visit['date'] = pd.to_datetime(df_visit['date'], format='%d %b %Y', errors='coerce').dt.date
df_visit.drop(columns=['date_time_start','date_time_end'],inplace=True)
df_visit['time_start'] = df_visit['time_start'].astype(str).str.strip()
df_visit['time_end'] = df_visit['time_end'].astype(str).str.strip()
df_visit['time_start'] = pd.to_datetime(df_visit['time_start'].astype(str), errors='coerce').dt.time
df_visit['time_end'] = pd.to_datetime(df_visit['time_end'].astype(str), errors='coerce').dt.time
df_visit['duration'] = (pd.to_datetime(df_visit['time_end'].astype(str), errors='coerce') - pd.to_datetime(df_visit['time_start'].astype(str), errors='coerce')).dt.total_seconds() / 60

def get_summary_data(pick_date="2024-11-01"):
    summary = df_visit[df_visit['date'] >= pd.to_datetime(pick_date).date()] if not df_visit.empty else pd.DataFrame()
    if not summary.empty:
        summary['lat'] = summary['lat'].astype(float)
        summary['long'] = summary['long'].astype(float)
        summary.reset_index(drop=True,inplace=True)
    data = []
    if not summary.empty:
        for dates in summary['date'].unique():
            for name in summary['employee_name'].unique():
                temp = summary[(summary.employee_name == name)&(summary['date'] == dates)].reset_index(drop=True)
                temp = temp[['date','employee_name','lat','long','time_start','time_end']]
                if len(temp) > 1:
                    dist = []
                    time_between = []
                    for i in range(len(temp)-1):
                        dist.append(round(geopy.distance.geodesic((temp.loc[i+1,'lat'],temp.loc[i+1,'long']), (temp.loc[i,'lat'],temp.loc[i,'long'])).km,2))
                        time_between.append((pd.to_datetime(str(temp.loc[i+1,'time_start'])) - pd.to_datetime(str(temp.loc[i,'time_start']))).total_seconds()/60)
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
        lat_long = summary[summary.employee_name == name][['lat','long']]
        min_lat = lat_long['lat'].min()
        max_lat = lat_long['lat'].max()
        min_long = lat_long['long'].min()
        max_long = lat_long['long'].max()
        lat_ = geopy.distance.geodesic((max_lat,min_long),(min_lat,min_long)).km
        long_ = geopy.distance.geodesic((min_lat,max_long),(min_lat,min_long)).km
        area = lat_ * long_
        filter_data.append([name,min_lat,max_lat,min_long,max_long,area])
area_coverage = pd.DataFrame(data=filter_data,columns=['employee_name','min_lat','max_lat','min_long','max_long','area']) if len(filter_data)>0 else pd.DataFrame(columns=['employee_name','min_lat','max_lat','min_long','max_long','area'])
for c in ['min_lat','min_long','max_lat','max_long']:
    if c in area_coverage.columns:
        area_coverage[c] = area_coverage[c].astype(float)

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
    get_dealer = df_dealer[(df_dealer.latitude.between(data_.min_lat.values[0],data_.max_lat.values[0]))&(df_dealer.longitude.between(data_.min_long.values[0],data_.max_long.values[0]))] if not df_dealer.empty else pd.DataFrame()
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
                    knee = KneeLocator(range(2, 2+len(valid_wcss)), valid_wcss, curve="convex", direction="decreasing")
                    n_cluster = knee.elbow if knee.elbow is not None else min(4, len(sum_))
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
        if not sum_.empty:
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

running_order = running_order.copy() if not running_order.empty else pd.DataFrame(columns=['Dealer Id','Dealer Name','IsActive','End Date','LMS Id'])
active_order = running_order[['Dealer Id','Dealer Name','IsActive','End Date']].copy() if 'Dealer Id' in running_order.columns else pd.DataFrame(columns=['Dealer Id','Dealer Name','IsActive','End Date'])
active_order = active_order[active_order.IsActive == "1"] if not active_order.empty else active_order
if not active_order.empty:
    active_order['End Date'] = pd.to_datetime(active_order['End Date'], errors='coerce')
    active_order['Dealer Id'] = pd.to_numeric(active_order['Dealer Id'], errors='coerce')
    active_order_group = active_order.groupby(['Dealer Id','Dealer Name']).agg({'End Date':'min'}).reset_index()
    active_order_group.rename(columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','End Date':'nearest_end_date'},inplace=True)
else:
    active_order_group = pd.DataFrame(columns=['id_dealer_outlet','dealer_name','nearest_end_date'])

run_order = running_order[['Dealer Id','Dealer Name','LMS Id','IsActive']].copy() if 'Dealer Id' in running_order.columns else pd.DataFrame(columns=['Dealer Id','Dealer Name','LMS Id','IsActive'])
run_order.rename(columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','LMS Id':'joined_dse','IsActive':'active_dse'},inplace=True)
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
    avail_df_merge = pd.merge(avail_df,run_order_group.drop(columns=['dealer_name']),how='left',on='id_dealer_outlet')
else:
    avail_df_merge = avail_df.copy() if not avail_df.empty else pd.DataFrame()

if not location_detail.empty:
    location_detail = location_detail.rename(columns={location_detail.columns[0]:'City', location_detail.columns[1]:'Cluster'}) if len(location_detail.columns)>=2 else location_detail
else:
    location_detail = pd.DataFrame(columns=['City','Cluster'])

if not avail_df_merge.empty and not location_detail.empty:
    avail_df_merge = pd.merge(avail_df_merge,location_detail[['City','Cluster']].rename(columns={'City':'city','Cluster':'cluster'}),how='left',on='city')
else:
    if 'city' not in avail_df_merge.columns:
        avail_df_merge['city'] = np.nan
    if 'cluster' not in avail_df_merge.columns:
        avail_df_merge['cluster'] = np.nan

cluster_left = cluster_left.copy() if not cluster_left.empty else pd.DataFrame(columns=['Category','Cluster','Brand','Daily_Gen','Daily_Need','Delta','Tag'])
if not cluster_left.empty:
    cluster_left = cluster_left.replace({'CHERY':'Chery','Kia':'KIA'})
    cluster_left = cluster_left.rename(columns={'Cluster':'cluster','Brand':'brand','Daily_Gen':'daily_gen','Daily_Need':'daily_need','Delta':'delta','Tag':'availability'})

if not avail_df_merge.empty and not cluster_left.empty:
    avail_df_merge = pd.merge(avail_df_merge, cluster_left[['cluster','brand','daily_gen','daily_need','delta','availability']], how='left', on=['brand','cluster'])
else:
    if 'availability' not in avail_df_merge.columns:
        avail_df_merge['availability'] = np.nan

avail_df_merge['tag'] = np.where(avail_df_merge.get('nearest_end_date').isna(),'Not Active', avail_df_merge.get('tag','Not Active'))

sales_data = sales_data.copy() if not sales_data.empty else pd.DataFrame(columns=['Date of Sales based on Image Proof','Sales ID','Amount'])
if not sales_data.empty:
    rev_data = sales_data[['Date of Sales based on Image Proof','Sales ID','Amount']].copy()
    rev_data.rename(columns={'Date of Sales based on Image Proof':'date','Sales ID':'sales_name','Amount':'amount'},inplace=True)
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
