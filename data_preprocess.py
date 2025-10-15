import pandas as pd
import numpy as np
import geopy.distance
from data_load import cluster_left, location_detail, df_visit, df_dealer, running_order
from sklearn.cluster import KMeans
from kneed import KneeLocator

df_dealer = df_dealer.copy()
df_dealer["business_type"] = "Car"
df_dealer = df_dealer[["id_dealer_outlet","brand","business_type","city","name","state","latitude","longitude"]]
df_dealer = df_dealer[df_dealer["business_type"].isin(["Car","Bike"])]
df_dealer = df_dealer.dropna().reset_index(drop=True)
df_dealer["latitude"] = df_dealer["latitude"].astype(str).str.replace(",","",regex=False).astype(float)
df_dealer["longitude"] = df_dealer["longitude"].astype(str).str.replace(",","",regex=False).str.strip(".").astype(float)

df_visit = df_visit.copy()
df_visit = df_visit[["Employee Name","Client Name","Date Time Start","Date Time End","Note Start","Note End","Longitude Start","Latitude Start"]]
df_visit = df_visit.rename(columns={"Employee Name":"employee_name","Client Name":"client_name","Date Time Start":"date_time_start","Date Time End":"date_time_end","Note Start":"note_start","Note End":"note_end","Longitude Start":"long","Latitude Start":"lat"})
df_visit["time_start"] = df_visit["date_time_start"].astype(str).apply(lambda x: x.split("@")[1] if "@" in x else np.nan)
df_visit["time_end"] = df_visit["date_time_end"].astype(str).apply(lambda x: x.split("@")[1] if "@" in x else np.nan)
df_visit["date"] = df_visit["date_time_start"].astype(str).apply(lambda x: x.split("@")[0] if "@" in x else np.nan)
df_visit["date"] = df_visit["date"].astype(str).str.strip()
df_visit["date"] = pd.to_datetime(df_visit["date"], format="%d %b %Y", errors="coerce").dt.date
df_visit = df_visit.drop(columns=["date_time_start","date_time_end"])
df_visit["time_start"] = df_visit["time_start"].astype(str).str.strip()
df_visit["time_end"] = df_visit["time_end"].astype(str).str.strip()
df_visit["time_start"] = pd.to_datetime(df_visit["time_start"], errors="coerce").dt.time
df_visit["time_end"] = pd.to_datetime(df_visit["time_end"], errors="coerce").dt.time
df_visit["duration"] = (pd.to_datetime(df_visit["time_end"].astype(str), errors="coerce") - pd.to_datetime(df_visit["time_start"].astype(str), errors="coerce")).dt.total_seconds() / 60
df_visit["lat"] = pd.to_numeric(df_visit["lat"], errors="coerce")
df_visit["long"] = pd.to_numeric(df_visit["long"], errors="coerce")
df_visit = df_visit.dropna(subset=["lat","long"]).reset_index(drop=True)

def get_summary_data(pick_date="2024-11-01"):
    summary = df_visit[df_visit["date"] >= pd.to_datetime(pick_date).date()].copy()
    summary["lat"] = summary["lat"].astype(float)
    summary["long"] = summary["long"].astype(float)
    summary = summary.reset_index(drop=True)
    data = []
    for dates in summary["date"].unique():
        for name in summary["employee_name"].unique():
            temp = summary[(summary.employee_name == name) & (summary["date"] == dates)].reset_index(drop=True)
            temp = temp[["date","employee_name","lat","long","time_start","time_end"]]
            if len(temp) > 1:
                dist = []
                time_between = []
                for i in range(len(temp)-1):
                    dist.append(round(geopy.distance.geodesic((temp.loc[i+1,"lat"], temp.loc[i+1,"long"]), (temp.loc[i,"lat"], temp.loc[i,"long"])).km, 2))
                    time_between.append((pd.to_datetime(str(temp.loc[i+1,"time_start"])) - pd.to_datetime(str(temp.loc[i,"time_start"]))).total_seconds()/60)
                avg_speed = round(sum(dist) / sum(time_between), 2) if sum(time_between) != 0 else 0
                data.append([dates, name, len(temp), round(np.mean(dist),2), round(np.mean(time_between),2), avg_speed])
            else:
                data.append([dates, name, len(temp), 0.0, 0.0, 0.0])
    cols = ["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"]
    data = pd.DataFrame(data, columns=cols)
    data["month_year"] = data["date"].astype(str).apply(lambda x: x.split("-")[0] + "-" + x.split("-")[1])
    return summary, data

summary, data_sum = get_summary_data()

filter_data = []
for name in summary.employee_name.unique():
    lat_long = summary[summary.employee_name == name][["lat","long"]]
    min_lat = lat_long["lat"].min()
    max_lat = lat_long["lat"].max()
    min_long = lat_long["long"].min()
    max_long = lat_long["long"].max()
    lat_ = geopy.distance.geodesic((max_lat, min_long), (min_lat, min_long)).km
    long_ = geopy.distance.geodesic((min_lat, max_long), (min_lat, min_long)).km
    area = lat_ * long_
    filter_data.append([name, min_lat, max_lat, min_long, max_long, area])

area_coverage = pd.DataFrame(data=filter_data, columns=["employee_name","min_lat","max_lat","min_long","max_long","area"])
area_coverage["min_lat"] = area_coverage["min_lat"].astype(float)
area_coverage["min_long"] = area_coverage["min_long"].astype(float)
area_coverage["max_lat"] = area_coverage["max_lat"].astype(float)
area_coverage["max_long"] = area_coverage["max_long"].astype(float)

def get_distance_dealer(cluster, lat, long):
    return geopy.distance.geodesic((kmeans.cluster_centers_[cluster,0], kmeans.cluster_centers_[cluster,1]), (lat, long)).km

sum_data = []
avail_data = []
cluster_center = []

for name in area_coverage.employee_name.unique():
    data_ = area_coverage[area_coverage.employee_name == name]
    get_dealer = df_dealer[(df_dealer.latitude.between(data_.min_lat.values[0], data_.max_lat.values[0])) & (df_dealer.longitude.between(data_.min_long.values[0], data_.max_long.values[0]))]
    sum_ = summary[summary.employee_name == name][["date","client_name","lat","long"]].rename(columns={"lat":"latitude","long":"longitude"})
    sum_["sales_name"] = name
    avail_ = get_dealer[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]]
    avail_["tag"] = "avail"
    avail_["sales_name"] = name
    if len(sum_) >= 2:
        wcss = []
        for i in range(4, min(9, len(sum_))):
            X = list(zip(sum_["latitude"], sum_["longitude"]))
            kmeans = KMeans(n_clusters=i).fit(X)
            wcss.append(kmeans.inertia_)
        knee = KneeLocator(range(4, min(9, len(sum_))), wcss, curve="convex", direction="decreasing")
        n_cluster = knee.elbow if knee.elbow is not None else 4
        kmeans = KMeans(n_clusters=n_cluster)
        data = list(zip(sum_["latitude"], sum_["longitude"]))
        kmeans.fit(data)
        sum_["cluster"] = kmeans.labels_
        for i in range(len(kmeans.cluster_centers_)):
            avail_[f"dist_center_{i}"] = avail_.apply(lambda x: get_distance_dealer(i, x.latitude, x.longitude), axis=1)
    else:
        sum_["cluster"] = 0
        kmeans = KMeans(n_clusters=1, n_init=1).fit(list(zip(sum_["latitude"], sum_["longitude"]))) if len(sum_) == 1 else KMeans(n_clusters=1, n_init=1).fit([(0.0,0.0)])
    clust_ = pd.DataFrame(kmeans.cluster_centers_, columns=["latitude","longitude"])
    clust_["sales_name"] = name
    clust_["cluster"] = range(len(kmeans.cluster_centers_))
    cluster_center.append(clust_)
    avail_data.append(avail_)
    sum_data.append(sum_)

sum_df = pd.concat(sum_data, ignore_index=True) if sum_data else pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
avail_df = pd.concat(avail_data, ignore_index=True) if avail_data else pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","latitude","longitude","tag","sales_name"])
clust_df = pd.concat(cluster_center, ignore_index=True) if cluster_center else pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])

active_order = running_order[["Dealer Id","Dealer Name","IsActive","End Date"]].copy()
active_order = active_order[active_order["IsActive"] == "1"]
active_order["End Date"] = pd.to_datetime(active_order["End Date"], errors="coerce")
active_order["Dealer Id"] = pd.to_numeric(active_order["Dealer Id"], errors="coerce")
active_order = active_order.dropna(subset=["Dealer Id"])
active_order["Dealer Id"] = active_order["Dealer Id"].astype(int)
ao_group = active_order.groupby(["Dealer Id","Dealer Name"], as_index=False)["End Date"].min().rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","End Date":"nearest_end_date"})

run_order = running_order[["Dealer Id","Dealer Name","LMS Id","IsActive"]].copy()
run_order = run_order.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
run_order["id_dealer_outlet"] = pd.to_numeric(run_order["id_dealer_outlet"], errors="coerce")
run_order["active_dse"] = pd.to_numeric(run_order["active_dse"], errors="coerce")
run_order = run_order.dropna(subset=["id_dealer_outlet"])
grouped_run_order = run_order.groupby(["id_dealer_outlet","dealer_name"], as_index=False).agg({"joined_dse":"count","active_dse":"sum"})
grouped_run_order = grouped_run_order.merge(ao_group[["id_dealer_outlet","nearest_end_date"]], on="id_dealer_outlet", how="left")
avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce")
avail_df = avail_df.dropna(subset=["id_dealer_outlet"])
avail_df["id_dealer_outlet"] = avail_df["id_dealer_outlet"].astype(int)

if avail_df.shape[1] > 9:
    min_values = avail_df.fillna(100000000).iloc[:, 9:].min(axis=1)
    avail_df.iloc[:, 9:] = avail_df.iloc[:, 9:].where(avail_df.iloc[:, 9:].eq(min_values, axis=0), np.nan)

avail_df_merge = avail_df.merge(grouped_run_order.drop(columns=["dealer_name"]), on="id_dealer_outlet", how="left")
ld = location_detail.rename(columns={"City":"city","Cluster":"cluster"})
avail_df_merge = avail_df_merge.merge(ld[["city","cluster"]], on="city", how="left")
cl = cluster_left.rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"})
cl = cl[cl.get("Category","Car") == "Car"].replace({"CHERY":"Chery","Kia":"KIA"})
avail_df_merge = avail_df_merge.merge(cl[["cluster","brand","daily_gen","daily_need","delta","availability"]], on=["brand","cluster"], how="left")
avail_df_merge["tag"] = np.where(avail_df_merge["nearest_end_date"].isna(), "Not Active", "Active")
