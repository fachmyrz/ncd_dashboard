import pandas as pd
import numpy as np
import geopy.distance
from data_load import *
from sklearn.cluster import KMeans
from kneed import KneeLocator
from datetime import datetime, timedelta


#Load Dealer Data
df_dealer = df_dealer.copy()

df_dealer['business_type'] = "Car"
df_dealer = df_dealer[['id_dealer_outlet','brand','business_type','city','name','state','latitude','longitude']]
df_dealer = df_dealer[df_dealer.business_type.isin(['Car','Bike'])]
df_dealer = df_dealer.dropna().reset_index(drop=True)
df_dealer['latitude'] = df_dealer['latitude'].str.replace(',', '', regex=False).astype(float)
df_dealer['longitude'] = df_dealer['longitude'].str.replace(',', '', regex=False).str.strip('.').astype(float)

#Load Client Visit Data
df_visit = df_visit.copy()

df_visit = df_visit[['Employee Name','Client Name','Date Time Start','Date Time End','Note Start','Note End','Longitude Start','Latitude Start']]
df_visit.rename(columns={'Employee Name':'employee_name','Client Name':'client_name','Date Time Start':'date_time_start','Date Time End':'date_time_end',
                         'Note Start':'note_start','Note End':'note_end','Longitude Start':'long','Latitude Start':'lat'},inplace=True)
df_visit['time_start'] = df_visit['date_time_start'].astype(str).apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
df_visit['time_end'] = df_visit['date_time_end'].astype(str).apply(lambda x: x.split('@')[1] if '@' in x else np.nan)
df_visit['date'] = df_visit['date_time_start'].astype(str).apply(lambda x: x.split('@')[0] if '@' in x else np.nan)

df_visit['date'] = df_visit['date'].str.strip()
df_visit['date'] = pd.to_datetime(df_visit['date'], format='%d %b %Y', errors='coerce').dt.date
df_visit.drop(columns=['date_time_start','date_time_end'],inplace=True)


df_visit['time_start'] = df_visit['time_start'].str.strip() # Remove leading/trailing spaces
df_visit['time_end'] = df_visit['time_end'].str.strip() # Remove leading/trailing spaces

df_visit['time_start'] = pd.to_datetime(df_visit['time_start'].astype(str)).dt.time
df_visit['time_end'] = pd.to_datetime(df_visit['time_end'].astype(str)).dt.time
# Calculate the time difference in seconds and then convert to minutes
df_visit['duration'] = (pd.to_datetime(df_visit['time_end'].astype(str)) - pd.to_datetime(df_visit['time_start'].astype(str))).dt.total_seconds() / 60


#Get Summary Data
def get_summary_data(pick_date="2024-11-01"):
    summary = df_visit[df_visit['date'] >= pd.to_datetime(pick_date).date()]
    summary['lat'] = summary['lat'].astype(float)
    summary['long'] = summary['long'].astype(float)
    summary.reset_index(drop=True,inplace=True)

    data = []
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
                
                # Avoid division by zero if sum(time_between) is 0
                avg_speed = round(sum(dist) / sum(time_between),2) if sum(time_between) != 0 else 0

                data.append([dates,name,len(temp),round(np.mean(dist),2),round(np.mean(time_between),2), avg_speed])
            else:
                dist = [0]
                time_between = [0]
                data.append([dates,name,len(temp),round(np.mean(dist),2),round(np.mean(time_between),2), 0]) # avg_speed is 0 when no movement

    cols = ['date','employee_name','ctd_visit','avg_distance_km','avg_time_between_minute','avg_speed_kmpm']
    data = pd.DataFrame(data,columns=cols)

    data['month_year'] = data['date'].astype(str).apply(lambda x: x.split('-')[0]+'-'+x.split('-')[1])

    return summary,data

summary,data_sum = get_summary_data()
#Area Coverage Eval
filter_data = []
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

area_coverage = pd.DataFrame(data=filter_data,columns=['employee_name','min_lat','max_lat','min_long','max_long','area'])
area_coverage['min_lat'] = area_coverage['min_lat'].astype(float)
area_coverage['min_long'] = area_coverage['min_long'].astype(float)
area_coverage['max_lat'] = area_coverage['max_lat'].astype(float)
area_coverage['max_long'] = area_coverage['max_long'].astype(float)

#Cluster Data
def get_distance_dealer(cluster,lat,long):
  return geopy.distance.geodesic((kmeans.cluster_centers_[cluster,0],kmeans.cluster_centers_[cluster,1]),(lat,long)).km

sum_data = []
avail_data = []
cluster_center = []

for name in area_coverage.employee_name.unique():
  #pilih area yang sesuai dengan orang
  data_ = area_coverage[area_coverage.employee_name == name]
  get_dealer = df_dealer[(df_dealer.latitude.between(data_.min_lat.values[0],data_.max_lat.values[0]))&(df_dealer.longitude.between(data_.min_long.values[0],data_.max_long.values[0]))]

  sum_ = summary[summary.employee_name == name][['date','client_name','lat','long']]
  sum_.rename(columns={'lat':'latitude','long':'longitude'},inplace=True)
  sum_['sales_name'] = name

  avail_ = get_dealer[['id_dealer_outlet','brand','business_type','city','name','latitude','longitude']]
  avail_['tag'] = 'avail'
  avail_['sales_name'] = name

  # Check if sum_ has enough data points for clustering
  if len(sum_) >= 2: # At least 2 data points are needed for meaningful clustering
    wcss = []
    for i in range(4, min(9, len(sum_))): # Limit clusters to the number of data points if less than 9
      X= list(zip(sum_['latitude'],sum_['longitude']))
      kmeans = KMeans(n_clusters=i).fit(X)
      wcss.append(kmeans.inertia_)

    knee = KneeLocator(range(4, min(9, len(sum_))), wcss, curve="convex", direction="decreasing")

    # Set a default n_cluster value if knee.elbow is None
    n_cluster = knee.elbow if knee.elbow is not None else 4  # or any other suitable default value

    kmeans = KMeans(n_clusters=n_cluster)

    data = list(zip(sum_['latitude'],sum_['longitude']))
    kmeans.fit(data)
    sum_['cluster'] = kmeans.labels_

    for i in range(len(kmeans.cluster_centers_)):
      avail_[f'dist_center_{i}'] = avail_.apply(lambda x: get_distance_dealer(i,x.latitude,x.longitude),axis=1)
  else:
    # Handle cases with insufficient data points (e.g., assign to a default cluster or skip)
    sum_['cluster'] = 0 # Assign to a default cluster 0
    #... other handling logic as needed

  clust_ = pd.DataFrame(kmeans.cluster_centers_,columns=['latitude','longitude'])
  clust_['sales_name'] = name
  clust_['cluster'] = range(len(kmeans.cluster_centers_))

  cluster_center.append(clust_)
  avail_data.append(avail_)
  sum_data.append(sum_)

#Result DF From Evaluation with Cluster
sum_df = pd.concat(sum_data)
avail_df = pd.concat(avail_data)
clust_df = pd.concat(cluster_center)

#Preprocess Active Order
active_order = running_order[['Dealer Id','Dealer Name','IsActive','End Date']]
active_order = active_order[active_order.IsActive == "1"]
active_order['End Date'] = pd.to_datetime(active_order['End Date'])
active_order['Dealer Id'] = active_order['Dealer Id'].astype(int)
active_order_group = active_order.groupby(['Dealer Id','Dealer Name']).agg({'End Date':'min'}).reset_index()
active_order_group.rename(columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','End Date':'nearest_end_date'},inplace=True)
active_order_group

#Preprocess Run Order
run_order = running_order[['Dealer Id','Dealer Name','LMS Id','IsActive']]
run_order.rename(columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','LMS Id':'joined_dse','IsActive':'active_dse'},inplace=True)

# Convert 'id_dealer_outlet' to numeric, handling errors
run_order['id_dealer_outlet'] = pd.to_numeric(run_order['id_dealer_outlet'], errors='coerce').astype('Int64') # Use errors='coerce' to handle non-numeric values
run_order['active_dse'] = pd.to_numeric(run_order['active_dse'], errors='coerce').astype('Int64')
run_order = run_order[~run_order.id_dealer_outlet.isna()]

# Group by and aggregate, ensuring 'nearest_end_date' is handled correctly
grouped_run_order = run_order.groupby(['id_dealer_outlet','dealer_name']).agg({
    'joined_dse':'count',
    'active_dse':'sum',
}).reset_index()

run_order_group = pd.merge(grouped_run_order,active_order_group,how='left',on=['id_dealer_outlet','dealer_name'])
run_order_group['id_dealer_outlet'] = run_order_group['id_dealer_outlet'].astype(int)
avail_df['id_dealer_outlet'] = avail_df['id_dealer_outlet'].astype(int)

# Find the minimum value for each row
min_values = avail_df.fillna(100000000).iloc[:, 9:].min(axis=1)

# Replace all values that are not the row minimum with NaN
avail_df.iloc[:, 9:] = avail_df.iloc[:, 9:].where(avail_df.iloc[:, 9:].eq(min_values, axis=0), np.nan)

#Merging Dataset
avail_df_merge = pd.merge(avail_df,run_order_group.drop(columns=['dealer_name']),how='left',on='id_dealer_outlet')
avail_df_merge = pd.merge(avail_df_merge,location_detail[['City','Cluster']].rename(columns={'City':'city','Cluster':'cluster'}),how='left',on='city')

avail_df_merge = pd.merge(avail_df_merge,
                  cluster_left[['Cluster','Brand','Daily_Gen','Daily_Need','Delta','Tag']][cluster_left.Category == "Car"].replace({'CHERY':'Chery','Kia':'KIA'}).rename(columns={'Cluster':'cluster',
                                                                      'Brand':'brand',
                                                                      'Daily_Gen':'daily_gen',
                                                                      'Daily_Need':'daily_need',
                                                                      'Delta':'delta',
                                                                      'Tag':'availability'}),how='left',on=['brand','cluster'])

avail_df_merge['tag'] = np.where(avail_df_merge.nearest_end_date.isna(),'Not Active','Active')

# #Revenue Data Process
# rev_data = sales_data[['Date of Sales based on Image Proof','Sales ID','Amount']]
# rev_data.rename(columns={'Date of Sales based on Image Proof':'date','Sales ID':'sales_name','Amount':'amount'},inplace=True)
# rev_data['date'] = pd.to_datetime(rev_data['date'])
# yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
# rev_data = rev_data[rev_data.date <= yesterday]

# # Replace commas in 'amount' column and then convert to int
# rev_data['amount'] = rev_data['amount'].str.replace(',', '', regex=False).astype(int)

# #Summazion Revenue
# gb_rev = rev_data[rev_data.date >= "2024-11-01"]
# gb_rev['month_year'] = gb_rev['date'].dt.to_period('M')
# # Convert 'month_year' to string before grouping
# gb_rev['month_year'] = gb_rev['month_year'].astype(str)
# gb_rev = gb_rev.groupby(['month_year','sales_name']).agg({'amount':'sum'}).reset_index()
