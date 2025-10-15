import pandas as pd
import numpy as np
import geopy.distance
from data_load import cluster_left, location_detail, df_visit, df_dealer, running_order
from sklearn.cluster import KMeans
from kneed import KneeLocator

def _clean_coord(s):
    return pd.to_numeric(
        pd.Series(s, dtype=str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9\.\-]+", "", regex=True)
        .str.replace(r"(?<!^)-", "", regex=True),
        errors="coerce",
    )

df_dealer = df_dealer.copy()
df_dealer["business_type"] = "Car"
df_dealer = df_dealer[["id_dealer_outlet","brand","business_type","city","name","state","latitude","longitude"]]
df_dealer = df_dealer.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
df_dealer["latitude"] = _clean_coord(df_dealer["latitude"])
df_dealer["longitude"] = _clean_coord(df_dealer["longitude"])
df_dealer = df_dealer[df_dealer["latitude"].between(-90,90) & df_dealer["longitude"].between(-180,180)].dropna(subset=["latitude","longitude"]).reset_index(drop=True)

df_visit = df_visit.copy()
if {"Employee Name","Client Name","Date Time Start","Date Time End"}.issubset(df_visit.columns):
    df_visit = df_visit[["Employee Name","Client Name","Date Time Start","Date Time End","Longitude Start","Latitude Start"]].rename(columns={"Employee Name":"employee_name","Client Name":"client_name","Date Time Start":"date_time_start","Date Time End":"date_time_end","Longitude Start":"long","Latitude Start":"lat"})
    df_visit["time_start"] = df_visit["date_time_start"].astype(str).apply(lambda x: x.split("@")[1] if "@" in x else np.nan)
    df_visit["time_end"] = df_visit["date_time_end"].astype(str).apply(lambda x: x.split("@")[1] if "@" in x else np.nan)
    df_visit["date"] = df_visit["date_time_start"].astype(str).apply(lambda x: x.split("@")[0] if "@" in x else np.nan)
    df_visit["date"] = df_visit["date"].astype(str).str.strip()
    df_visit["date"] = pd.to_datetime(df_visit["date"], format="%d %b %Y", errors="coerce").dt.date
else:
    for c_from, c_to in [("Nama Karyawan","employee_name"),("Nama Klien","client_name"),("Tanggal Datang","visit_dt"),("Latitude & Longitude Datang","ll")]:
        if c_from in df_visit.columns:
            df_visit = df_visit.rename(columns={c_from:c_to})
    if "visit_dt" in df_visit.columns:
        df_visit["date"] = pd.to_datetime(df_visit["visit_dt"], errors="coerce").dt.date
        df_visit["time_start"] = pd.to_datetime(df_visit["visit_dt"], errors="coerce").dt.time
        df_visit["time_end"] = np.nan
    if "ll" in df_visit.columns:
        parts = df_visit["ll"].astype(str).str.split(",", n=1, expand=True)
        df_visit["lat"] = _clean_coord(parts[0])
        df_visit["long"] = _clean_coord(parts[1] if parts.shape[1] > 1 else np.nan)
    else:
        if "Latitude Start" in df_visit.columns and "Longitude Start" in df_visit.columns:
            df_visit["lat"] = _clean_coord(df_visit["Latitude Start"])
            df_visit["long"] = _clean_coord(df_visit["Longitude Start"])
        else:
            df_visit["lat"] = np.nan
            df_visit["long"] = np.nan

if "lat" not in df_visit.columns or "long" not in df_visit.columns:
    df_visit["lat"] = np.nan
    df_visit["long"] = np.nan
df_visit["time_start"] = pd.to_datetime(df_visit.get("time_start", np.nan), errors="coerce").dt.time
df_visit["time_end"] = pd.to_datetime(df_visit.get("time_end", np.nan), errors="coerce").dt.time
df_visit["duration"] = (pd.to_datetime(df_visit["time_end"].astype(str), errors="coerce") - pd.to_datetime(df_visit["time_start"].astype(str), errors="coerce")).dt.total_seconds() / 60
df_visit["lat"] = _clean_coord(df_visit["lat"])
df_visit["long"] = _clean_coord(df_visit["long"])
df_visit = df_visit[df_visit["lat"].between(-90,90) & df_visit["long"].between(-180,180)].reset_index(drop=True)

def get_summary_data(pick_date="2024-11-01"):
    summary = df_visit[df_visit["date"] >= pd.to_datetime(pick_date).date()].copy()
    summary["lat"] = summary["lat"].astype(float)
    summary["long"] = summary["long"].astype(float)
    summary = summary.reset_index(drop=True)
    data = []
    for dates in summary["date"].dropna().unique():
        for name in summary["employee_name"].dropna().unique():
            temp = summary[(summary.employee_name == name) & (summary["date"] == dates)].reset_index(drop=True)
            temp = temp[["date","employee_name","lat","long","time_start","time_end"]]
            if len(temp) > 1:
                dist = []
                time_between = []
                for i in range(len(temp)-1):
                    try:
                        d = geopy.distance.geodesic((temp.loc[i+1,"lat"], temp.loc[i+1,"long"]), (temp.loc[i,"lat"], temp.loc[i,"long"])).km
                    except Exception:
                        d = 0.0
                    dist.append(round(d,2))
                    try:
                        tb = (pd.to_datetime(str(temp.loc[i+1,"time_start"])) - pd.to_datetime(str(temp.loc[i,"time_start"]))).total_seconds()/60
                    except Exception:
                        tb = 0.0
                    time_between.append(tb)
                avg_speed = round((sum(dist) / sum([t for t in time_between if t > 0])) if any(t > 0 for t in time_between) else 0, 2)
                data.append([dates, name, len(temp), round(np.mean(dist),2) if dist else 0.0, round(np.mean(time_between),2) if time_between else 0.0, avg_speed])
            else:
                data.append([dates, name, len(temp), 0.0, 0.0, 0.0])
    cols = ["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"]
    data = pd.DataFrame(data, columns=cols) if data else pd.DataFrame(columns=cols)
    if not data.empty:
        data["month_year"] = data["date"].astype(str).str.slice(0,7)
    else:
        data["month_year"] = pd.Series(dtype=str)
    return summary, data

summary, data_sum = get_summary_data()

filter_data = []
for name in summary["employee_name"].dropna().unique():
    lat_long = summary[summary.employee_name == name][["lat","long"]]
    if lat_long.empty:
        continue
    min_lat = lat_long["lat"].min()
    max_lat = lat_long["lat"].max()
    min_long = lat_long["long"].min()
    max_long = lat_long["long"].max()
    try:
        lat_ = geopy.distance.geodesic((max_lat, min_long), (min_lat, min_long)).km
        long_ = geopy.distance.geodesic((min_lat, max_long), (min_lat, min_long)).km
        area = lat_ * long_
    except Exception:
        area = 0.0
    filter_data.append([name, min_lat, max_lat, min_long, max_long, area])

area_coverage = pd.DataFrame(data=filter_data, columns=["employee_name","min_lat","max_lat","min_long","max_long","area"])

def _distance_centroid(centers, cluster, lat, lon):
    c = centers[cluster]
    return geopy.distance.geodesic((c[0], c[1]), (lat, lon)).km

sum_data = []
avail_data = []
cluster_center = []

for name in area_coverage["employee_name"].dropna().unique():
    data_ = area_coverage[area_coverage.employee_name == name]
    if data_.empty:
        continue
    get_dealer = df_dealer[(df_dealer.latitude.between(data_.min_lat.values[0], data_.max_lat.values[0])) & (df_dealer.longitude.between(data_.min_long.values[0], data_.max_long.values[0]))]
    sum_ = summary[summary.employee_name == name][["date","client_name","lat","long"]].rename(columns={"lat":"latitude","long":"longitude"})
    if sum_.empty:
        continue
    sum_["sales_name"] = name
    avail_ = get_dealer[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
    avail_["tag"] = "avail"
    avail_["sales_name"] = name
    if len(sum_) >= 4:
        X = list(zip(sum_["latitude"], sum_["longitude"]))
        wcss = []
        k_range = list(range(4, min(9, len(sum_)) + 1))
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            wcss.append(km.inertia_)
        knee = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
        n_cluster = knee.elbow if knee.elbow is not None else min(4, len(sum_))
        kmeans = KMeans(n_clusters=n_cluster, n_init=10, random_state=42).fit(X)
        centers = kmeans.cluster_centers_
        sum_["cluster"] = kmeans.labels_
        for i in range(len(centers)):
            avail_[f"dist_center_{i}"] = avail_.apply(lambda x: _distance_centroid(centers, i, x.latitude, x.longitude), axis=1)
    else:
        centers = np.array([[sum_["latitude"].mean(), sum_["longitude"].mean()]])
        sum_["cluster"] = 0
        avail_["dist_center_0"] = avail_.apply(lambda x: _distance_centroid(centers, 0, x.latitude, x.longitude), axis=1)
    clust_ = pd.DataFrame(centers, columns=["latitude","longitude"])
    clust_["sales_name"] = name
    clust_["cluster"] = range(len(centers))
    cluster_center.append(clust_)
    avail_data.append(avail_)
    sum_data.append(sum_)

sum_df = pd.concat(sum_data, ignore_index=True) if sum_data else pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
avail_df = pd.concat(avail_data, ignore_index=True) if avail_data else pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","latitude","longitude","tag","sales_name"])
clust_df = pd.concat(cluster_center, ignore_index=True) if cluster_center else pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])

active_order = running_order[["Dealer Id","Dealer Name","IsActive","End Date"]].copy() if "Dealer Id" in running_order.columns else pd.DataFrame(columns=["Dealer Id","Dealer Name","IsActive","End Date"])
active_order = active_order[active_order["IsActive"] == "1"] if not active_order.empty else active_order
active_order["End Date"] = pd.to_datetime(active_order.get("End Date", pd.Series(dtype=str)), errors="coerce")
active_order["Dealer Id"] = pd.to_numeric(active_order.get("Dealer Id", pd.Series(dtype=str)), errors="coerce")
active_order = active_order.dropna(subset=["Dealer Id"])
active_order["Dealer Id"] = active_order["Dealer Id"].astype(int)
ao_group = active_order.groupby(["Dealer Id","Dealer Name"], as_index=False)["End Date"].min().rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","End Date":"nearest_end_date"}) if not active_order.empty else pd.DataFrame(columns=["id_dealer_outlet","dealer_name","nearest_end_date"])

run_order = running_order[["Dealer Id","Dealer Name","LMS Id","IsActive"]].copy() if "Dealer Id" in running_order.columns else pd.DataFrame(columns=["Dealer Id","Dealer Name","LMS Id","IsActive"])
run_order = run_order.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
run_order["id_dealer_outlet"] = pd.to_numeric(run_order.get("id_dealer_outlet", pd.Series(dtype=str)), errors="coerce")
run_order["active_dse"] = pd.to_numeric(run_order.get("active_dse", pd.Series(dtype=str)), errors="coerce")
run_order = run_order.dropna(subset=["id_dealer_outlet"])
grouped_run_order = run_order.groupby(["id_dealer_outlet","dealer_name"], as_index=False).agg({"joined_dse":"count","active_dse":"sum"}) if not run_order.empty else pd.DataFrame(columns=["id_dealer_outlet","dealer_name","joined_dse","active_dse"])
if not ao_group.empty and not grouped_run_order.empty:
    grouped_run_order = grouped_run_order.merge(ao_group[["id_dealer_outlet","nearest_end_date"]], on="id_dealer_outlet", how="left")
else:
    if "nearest_end_date" not in grouped_run_order.columns:
        grouped_run_order["nearest_end_date"] = pd.NaT

avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df.get("id_dealer_outlet", pd.Series(dtype=str)), errors="coerce")
avail_df = avail_df.dropna(subset=["id_dealer_outlet"])
avail_df["id_dealer_outlet"] = avail_df["id_dealer_outlet"].astype(int)

dist_cols = [c for c in avail_df.columns if c.startswith("dist_center_")]
if dist_cols:
    min_values = avail_df[dist_cols].fillna(1e12).min(axis=1)
    for c in dist_cols:
        mask_min = avail_df[c] == min_values
        avail_df.loc[~mask_min, c] = np.nan

ld = location_detail.rename(columns={"City":"city","Cluster":"cluster"}) if not location_detail.empty else pd.DataFrame(columns=["city","cluster"])
avail_df_merge = avail_df.merge(grouped_run_order.drop(columns=["dealer_name"], errors="ignore"), on="id_dealer_outlet", how="left")
if not ld.empty:
    avail_df_merge = avail_df_merge.merge(ld[["city","cluster"]], on="city", how="left")

cl = cluster_left.rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"}) if not cluster_left.empty else pd.DataFrame(columns=["cluster","brand","daily_gen","daily_need","delta","availability","Category"])
if "Category" in cl.columns:
    cl = cl[cl["Category"] == "Car"].copy()
cl = cl.replace({"CHERY":"Chery","Kia":"KIA"})
if not cl.empty:
    avail_df_merge = avail_df_merge.merge(cl[["cluster","brand","daily_gen","daily_need","delta","availability"]], on=["brand","cluster"], how="left")

avail_df_merge["tag"] = np.where(avail_df_merge.get("nearest_end_date").isna(), "Not Active", "Active")
