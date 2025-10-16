import pandas as pd
import numpy as np
import geopy.distance
from data_load import *
from sklearn.cluster import KMeans
from kneed import KneeLocator

df_dealer = df_dealer.copy()
df_dealer = df_dealer[['id_dealer_outlet','brand','business_type','city','name','state','latitude','longitude']]
df_dealer = df_dealer.dropna().reset_index(drop=True)
df_dealer['business_type'] = df_dealer['business_type'].astype(str).str.strip()
df_dealer = df_dealer[df_dealer['business_type'].str.lower() == 'car']
df_dealer['latitude'] = (
    df_dealer['latitude'].astype(str)
    .str.replace('`','',regex=False)
    .str.replace(',','',regex=False)
    .str.strip()
    .str.strip('.')
).astype(float)
df_dealer['longitude'] = (
    df_dealer['longitude'].astype(str)
    .str.replace('`','',regex=False)
    .str.replace(',','',regex=False)
    .str.strip()
    .str.strip('.')
).astype(float)

df_visit = df_visit.copy()
df_visit = df_visit[['Employee Name','Client Name','Date Time Start','Date Time End','Note Start','Note End','Longitude Start','Latitude Start']]
df_visit.rename(columns={
    'Employee Name':'employee_name',
    'Client Name':'client_name',
    'Date Time Start':'date_time_start',
    'Date Time End':'date_time_end',
    'Note Start':'note_start',
    'Note End':'note_end',
    'Longitude Start':'long',
    'Latitude Start':'lat'
},inplace=True)
df_visit['time_start'] = df_visit['date_time_start'].astype(str).apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
df_visit['time_end'] = df_visit['date_time_end'].astype(str).apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
df_visit['date'] = df_visit['date_time_start'].astype(str).apply(lambda x: x.split('@')[0] if '@' in x else np.nan)
df_visit['date'] = df_visit['date'].str.strip()
df_visit['date'] = pd.to_datetime(df_visit['date'], format='%d %b %Y', errors='coerce').dt.date
df_visit.drop(columns=['date_time_start','date_time_end'],inplace=True)
df_visit['time_start'] = df_visit['time_start'].astype(str).str.strip()
df_visit['time_end'] = df_visit['time_end'].astype(str).str.strip()
df_visit['time_start'] = pd.to_datetime(df_visit['time_start'], errors='coerce').dt.time
df_visit['time_end'] = pd.to_datetime(df_visit['time_end'], errors='coerce').dt.time
df_visit['duration'] = (
    pd.to_datetime(df_visit['time_end'].astype(str), errors='coerce') -
    pd.to_datetime(df_visit['time_start'].astype(str), errors='coerce')
).dt.total_seconds() / 60

def get_summary_data(pick_date="2024-11-01"):
    summary = df_visit[df_visit['date'] >= pd.to_datetime(pick_date).date()].copy()
    summary['lat'] = pd.to_numeric(summary['lat'], errors='coerce')
    summary['long'] = pd.to_numeric(summary['long'], errors='coerce')
    summary = summary.dropna(subset=['lat','long'])
    summary.reset_index(drop=True,inplace=True)
    data = []
    for dates in summary['date'].dropna().unique():
        for name in summary['employee_name'].dropna().unique():
            temp = summary[(summary.employee_name == name)&(summary['date'] == dates)].reset_index(drop=True)
            temp = temp[['date','employee_name','lat','long','time_start','time_end']]
            if len(temp) > 1:
                dist = []
                time_between = []
                for i in range(len(temp)-1):
                    dist.append(round(geopy.distance.geodesic((temp.loc[i+1,'lat'],temp.loc[i+1,'long']), (temp.loc[i,'lat'],temp.loc[i,'long'])).km,2))
                    ts1 = pd.to_datetime(str(temp.loc[i,'time_start']), errors='coerce')
                    ts2 = pd.to_datetime(str(temp.loc[i+1,'time_start']), errors='coerce')
                    tb = (ts2 - ts1).total_seconds()/60 if ts1==ts1 and ts2==ts2 else 0
                    time_between.append(tb)
                avg_speed = round(sum(dist) / sum(time_between),2) if sum(time_between) != 0 else 0
                data.append([dates,name,len(temp),round(np.mean(dist),2),round(np.mean(time_between),2), avg_speed])
            else:
                data.append([dates,name,len(temp),0.0,0.0,0.0])
    cols = ['date','employee_name','ctd_visit','avg_distance_km','avg_time_between_minute','avg_speed_kmpm']
    data = pd.DataFrame(data,columns=cols)
    data['month_year'] = pd.to_datetime(data['date']).dt.to_period('M').astype(str)
    return summary,data

summary,data_sum = get_summary_data()

filter_data = []
for name in summary['employee_name'].dropna().unique():
  lat_long = summary[summary.employee_name == name][['lat','long']].dropna()
  if lat_long.empty:
      continue
  min_lat = lat_long['lat'].min()
  max_lat = lat_long['lat'].max()
  min_long = lat_long['long'].min()
  max_long = lat_long['long'].max()
  lat_ = geopy.distance.geodesic((max_lat,min_long),(min_lat,min_long)).km
  long_ = geopy.distance.geodesic((min_lat,max_long),(min_lat,min_long)).km
  area = lat_ * long_
  filter_data.append([name,min_lat,max_lat,min_long,max_long,area])

area_coverage = pd.DataFrame(data=filter_data,columns=['employee_name','min_lat','max_lat','min_long','max_long','area'])
for c in ['min_lat','min_long','max_lat','max_long']:
    area_coverage[c] = pd.to_numeric(area_coverage[c], errors='coerce')

def get_distance_dealer(cluster,lat,long):
  return geopy.distance.geodesic((kmeans.cluster_centers_[cluster,0],kmeans.cluster_centers_[cluster,1]),(lat,long)).km

sum_data = []
avail_data = []
cluster_center = []

for name in area_coverage['employee_name'].dropna().unique():
  data_ = area_coverage[area_coverage.employee_name == name]
  if data_.empty:
      continue
  get_dealer = df_dealer[
      (df_dealer.latitude.between(data_.min_lat.values[0],data_.max_lat.values[0]))&
      (df_dealer.longitude.between(data_.min_long.values[0],data_.max_long.values[0]))
  ]
  sum_ = summary[summary.employee_name == name][['date','client_name','lat','long']].copy()
  sum_.rename(columns={'lat':'latitude','long':'longitude'},inplace=True)
  sum_['sales_name'] = name

  avail_ = get_dealer[['id_dealer_outlet','brand','business_type','city','name','latitude','longitude']].copy()
  avail_['tag'] = 'avail'
  avail_['sales_name'] = name

  if len(sum_) >= 2:
    wcss = []
    cand = list(range(4, min(9, len(sum_)+1)))
    if not cand:
        cand = [2,3,4]
    for i in cand:
      X= list(zip(sum_['latitude'],sum_['longitude']))
      kmeans = KMeans(n_clusters=i, n_init=10).fit(X)
      wcss.append(kmeans.inertia_)
    start = 4 if 4 in cand else cand[0]
    knee = KneeLocator(range(start, start+len(wcss)), wcss, curve="convex", direction="decreasing")
    n_cluster = knee.elbow if knee.elbow is not None else min(4, len(sum_)) if len(sum_)>=4 else 2
    kmeans = KMeans(n_clusters=n_cluster, n_init=10)
    data_xy = list(zip(sum_['latitude'],sum_['longitude']))
    kmeans.fit(data_xy)
    sum_['cluster'] = kmeans.labels_
    for i in range(len(kmeans.cluster_centers_)):
      avail_[f'dist_center_{i}'] = avail_.apply(lambda x: get_distance_dealer(i,x.latitude,x.longitude),axis=1)
  else:
    n_cluster = 1
    kmeans = KMeans(n_clusters=n_cluster, n_init=10).fit(list(zip(sum_['latitude'],sum_['longitude']))) if len(sum_)==1 else KMeans(n_clusters=1, n_init=10).fit([[0,0],[0,0]])
    sum_['cluster'] = 0

  clust_ = pd.DataFrame(kmeans.cluster_centers_,columns=['latitude','longitude'])
  clust_['sales_name'] = name
  clust_['cluster'] = range(len(kmeans.cluster_centers_))
  cluster_center.append(clust_)
  avail_data.append(avail_)
  sum_data.append(sum_)

sum_df = pd.concat(sum_data) if sum_data else pd.DataFrame(columns=['date','client_name','latitude','longitude','sales_name','cluster'])
avail_df = pd.concat(avail_data) if avail_data else pd.DataFrame(columns=['id_dealer_outlet','brand','business_type','city','name','latitude','longitude','tag','sales_name'])
clust_df = pd.concat(cluster_center) if cluster_center else pd.DataFrame(columns=['latitude','longitude','sales_name','cluster'])

active_order = running_order[['Dealer Id','Dealer Name','IsActive','End Date']]
active_order = active_order[active_order.IsActive == "1"]
active_order['End Date'] = pd.to_datetime(active_order['End Date'], errors='coerce')
active_order['Dealer Id'] = pd.to_numeric(active_order['Dealer Id'], errors='coerce').astype('Int64')
active_order_group = active_order.groupby(['Dealer Id','Dealer Name']).agg({'End Date':'min'}).reset_index()
active_order_group.rename(columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','End Date':'nearest_end_date'},inplace=True)

run_order = running_order[['Dealer Id','Dealer Name','LMS Id','IsActive']].copy()
run_order.rename(columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','LMS Id':'joined_dse','IsActive':'active_dse'},inplace=True)
run_order['id_dealer_outlet'] = pd.to_numeric(run_order['id_dealer_outlet'], errors='coerce').astype('Int64')
run_order['active_dse'] = pd.to_numeric(run_order['active_dse'], errors='coerce').astype('Int64')
run_order = run_order[~run_order['id_dealer_outlet'].isna()]
grouped_run_order = run_order.groupby(['id_dealer_outlet','dealer_name']).agg({
    'joined_dse':'count',
    'active_dse':'sum',
}).reset_index()
run_order_group = pd.merge(grouped_run_order,active_order_group,how='left',on=['id_dealer_outlet','dealer_name'])
run_order_group['id_dealer_outlet'] = pd.to_numeric(run_order_group['id_dealer_outlet'], errors='coerce').astype('Int64')

avail_df['id_dealer_outlet'] = pd.to_numeric(avail_df['id_dealer_outlet'], errors='coerce').astype('Int64')

start_idx = 0
for i,c in enumerate(avail_df.columns):
    if str(c).startswith('dist_center_'):
        start_idx = i
        break
if start_idx>0:
    min_values = avail_df.iloc[:, start_idx:].apply(pd.to_numeric, errors='coerce').fillna(1e12).min(axis=1)
    mask = avail_df.iloc[:, start_idx:].apply(pd.to_numeric, errors='coerce').eq(min_values, axis=0)
    avail_df.iloc[:, start_idx:] = avail_df.iloc[:, start_idx:].where(mask, np.nan)

avail_df_merge = pd.merge(avail_df, run_order_group.drop(columns=['dealer_name']), how='left', on='id_dealer_outlet')
ld = location_detail[['City','Cluster']].rename(columns={'City':'city','Cluster':'cluster'})
avail_df_merge = pd.merge(avail_df_merge, ld, how='left', on='city')

cl_map = cluster_left.copy()
cl_map = cl_map.replace({'CHERY':'Chery','Kia':'KIA'})
cl_map = cl_map[cl_map['Category'].astype(str).str.lower() == 'car']
cl_map = cl_map[['Cluster','Brand','Daily_Gen','Daily_Need','Delta','Tag']].rename(columns={
    'Cluster':'cluster',
    'Brand':'brand',
    'Daily_Gen':'daily_gen',
    'Daily_Need':'daily_need',
    'Delta':'delta',
    'Tag':'availability'
})
avail_df_merge = pd.merge(avail_df_merge, cl_map, how='left', on=['brand','cluster'])
avail_df_merge['tag'] = np.where(avail_df_merge['nearest_end_date'].isna(),'Not Active','Active')
