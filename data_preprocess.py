import pandas as pd
import numpy as np
import geopy.distance
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from kneed import KneeLocator
from data_load import cluster_left, location_detail, df_visit, df_dealer, running_order

pd.options.mode.copy_on_write = True

def _to_float_series(s):
    return pd.to_numeric(pd.Series(s, dtype="object").astype(str).str.replace(",", "", regex=False).str.replace("`", "", regex=False).str.strip(), errors="coerce")

def _before_at(x):
    x = "" if x is None else str(x)
    return x.split("@")[0] if "@" in x else x

def _after_at(x):
    x = "" if x is None else str(x)
    return x.split("@")[1].strip() if "@" in x else None

df_dealer = df_dealer.copy()
df_dealer["business_type"] = "Car"
df_dealer = df_dealer[["id_dealer_outlet","brand","business_type","city","name","state","latitude","longitude"]]
df_dealer = df_dealer[df_dealer["business_type"].isin(["Car"])]
df_dealer["latitude"] = _to_float_series(df_dealer["latitude"])
df_dealer["longitude"] = _to_float_series(df_dealer["longitude"])
df_dealer = df_dealer.dropna(subset=["latitude","longitude"]).reset_index(drop=True)

df_visit = df_visit.copy()
rename_map = {}
for c in df_visit.columns:
    lc = str(c).lower().strip()
    if lc in ["employee name","nama karyawan"]:
        rename_map[c] = "employee_name"
    if lc in ["client name","nama klien"]:
        rename_map[c] = "client_name"
    if lc in ["date time start","tanggal datang"]:
        rename_map[c] = "date_time_start"
    if lc in ["date time end"]:
        rename_map[c] = "date_time_end"
    if lc in ["note start"]:
        rename_map[c] = "note_start"
    if lc in ["note end"]:
        rename_map[c] = "note_end"
    if lc in ["longitude start","longitude datang"]:
        rename_map[c] = "long"
    if lc in ["latitude start","latitude datang"]:
        rename_map[c] = "lat"
    if lc == "nomor induk karyawan":
        rename_map[c] = "nomor_induk_karyawan"
    if lc == "divisi":
        rename_map[c] = "divisi"
df_visit = df_visit.rename(columns=rename_map)
for k in ["employee_name","client_name","date_time_start","date_time_end","note_start","note_end","long","lat","nomor_induk_karyawan","divisi"]:
    if k not in df_visit.columns:
        df_visit[k] = np.nan
df_visit["date"] = pd.to_datetime(df_visit["date_time_start"].map(_before_at), errors="coerce").dt.date
df_visit["time_start"] = pd.to_datetime(df_visit["date_time_start"].map(_after_at), errors="coerce").dt.time
df_visit["time_end"] = pd.to_datetime(df_visit["date_time_end"].map(_after_at), errors="coerce").dt.time
df_visit["lat"] = _to_float_series(df_visit["lat"])
df_visit["long"] = _to_float_series(df_visit["long"])
df_visit["duration"] = (pd.to_datetime(df_visit["time_end"].astype(str), errors="coerce") - pd.to_datetime(df_visit["time_start"].astype(str), errors="coerce")).dt.total_seconds() / 60

def get_summary_data(vis=None, pick_date="2024-11-01"):
    if vis is None:
        summary = df_visit[df_visit["date"] >= pd.to_datetime(pick_date).date()].copy()
    else:
        summary = vis[vis["date"] >= pd.to_datetime(pick_date).date()].copy()
    summary["lat"] = pd.to_numeric(summary["lat"], errors="coerce")
    summary["long"] = pd.to_numeric(summary["long"], errors="coerce")
    summary = summary.dropna(subset=["lat","long"]).reset_index(drop=True)
    data = []
    if summary.empty:
        out = pd.DataFrame(columns=["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"])
        out["month_year"] = pd.Series(dtype="object")
        return summary, out
    for dates in summary["date"].unique():
        for name in summary["employee_name"].dropna().unique():
            temp = summary[(summary.employee_name == name) & (summary["date"] == dates)].reset_index(drop=True)
            temp = temp[["date","employee_name","lat","long","time_start","time_end"]]
            if len(temp) > 1:
                dist = []
                time_between = []
                for i in range(len(temp) - 1):
                    a = (temp.loc[i + 1, "lat"], temp.loc[i + 1, "long"])
                    b = (temp.loc[i, "lat"], temp.loc[i, "long"])
                    d = geopy.distance.geodesic(a, b).km
                    dist.append(round(d, 2))
                    tb = (pd.to_datetime(str(temp.loc[i + 1, "time_start"])) - pd.to_datetime(str(temp.loc[i, "time_start"]))).total_seconds() / 60
                    time_between.append(tb)
                sp = round((sum(dist) / sum(time_between)), 2) if sum(time_between) != 0 else 0
                data.append([dates, name, len(temp), round(np.mean(dist), 2), round(np.mean(time_between), 2), sp])
            else:
                data.append([dates, name, 1, 0.0, 0.0, 0.0])
    cols = ["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"]
    data = pd.DataFrame(data, columns=cols)
    data["month_year"] = pd.to_datetime(data["date"], errors="coerce").dt.to_period("M").astype(str)
    return summary, data

summary, data_sum = get_summary_data()

area_rows = []
if not summary.empty:
    for name in summary.employee_name.dropna().unique():
        ll = summary[summary.employee_name == name][["lat","long"]].dropna()
        if ll.empty:
            continue
        min_lat = ll["lat"].min()
        max_lat = ll["lat"].max()
        min_long = ll["long"].min()
        max_long = ll["long"].max()
        lat_km = geopy.distance.geodesic((max_lat, min_long), (min_lat, min_long)).km
        lon_km = geopy.distance.geodesic((min_lat, max_long), (min_lat, min_long)).km
        area = lat_km * lon_km
        area_rows.append([name, min_lat, max_lat, min_long, max_long, area])
area_coverage = pd.DataFrame(area_rows, columns=["employee_name","min_lat","max_lat","min_long","max_long","area"])
for c in ["min_lat","max_lat","min_long","max_long"]:
    if c in area_coverage.columns:
        area_coverage[c] = pd.to_numeric(area_coverage[c], errors="coerce")

def _kmeans_labels(df_xy):
    if len(df_xy) < 2:
        return np.zeros(len(df_xy), dtype=int), np.array([[df_xy["latitude"].mean(), df_xy["longitude"].mean()]])
    wcss = []
    max_k = min(8, max(4, len(df_xy)))
    X = list(zip(df_xy["latitude"], df_xy["longitude"]))
    for i in range(4, max_k + 1):
        km = KMeans(n_clusters=i, n_init="auto").fit(X)
        wcss.append(km.inertia_)
    knee = KneeLocator(range(4, max_k + 1), wcss, curve="convex", direction="decreasing")
    n_cluster = knee.elbow if getattr(knee, "elbow", None) else min(4, max_k)
    km = KMeans(n_clusters=n_cluster, n_init="auto")
    km.fit(X)
    return km.labels_, km.cluster_centers_

sum_data = []
avail_data = []
cluster_center_list = []

names_iter = area_coverage["employee_name"].dropna().unique() if not area_coverage.empty else []
for name in names_iter:
    bounds = area_coverage[area_coverage.employee_name == name].iloc[0]
    dealers_in = df_dealer[df_dealer.latitude.between(bounds["min_lat"], bounds["max_lat"]) & df_dealer.longitude.between(bounds["min_long"], bounds["max_long"])].copy()
    s = summary[summary.employee_name == name][["date","client_name","lat","long"]].rename(columns={"lat":"latitude","long":"longitude"}).copy()
    s["sales_name"] = name
    a = dealers_in[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
    a["tag"] = "avail"
    a["sales_name"] = name
    if len(s) >= 2:
        labs, centers = _kmeans_labels(s)
        s["cluster"] = labs
        for i in range(len(centers)):
            c_lat, c_lon = centers[i, 0], centers[i, 1]
            a[f"dist_center_{i}"] = a.apply(lambda r: geopy.distance.geodesic((c_lat, c_lon), (r.latitude, r.longitude)).km, axis=1)
    else:
        s["cluster"] = 0
        centers = np.array([[s["latitude"].mean() if not s.empty else dealers_in["latitude"].mean(), s["longitude"].mean() if not s.empty else dealers_in["longitude"].mean()]])
        a["dist_center_0"] = a.apply(lambda r: geopy.distance.geodesic((centers[0, 0], centers[0, 1]), (r.latitude, r.longitude)).km, axis=1)
    cl = pd.DataFrame(centers, columns=["latitude","longitude"])
    cl["sales_name"] = name
    cl["cluster"] = range(len(centers))
    cluster_center_list.append(cl)
    avail_data.append(a)
    sum_data.append(s)

sum_df = pd.concat(sum_data).reset_index(drop=True) if sum_data else pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
avail_df = pd.concat(avail_data).reset_index(drop=True) if avail_data else pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","latitude","longitude","tag","sales_name","dist_center_0"])
clust_df = pd.concat(cluster_center_list).reset_index(drop=True) if cluster_center_list else pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])

active_order = running_order.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","IsActive":"is_active","End Date":"end_date"})
active_order["end_date"] = pd.to_datetime(active_order["end_date"], errors="coerce")
active_order["id_dealer_outlet"] = pd.to_numeric(active_order["id_dealer_outlet"], errors="coerce").astype("Int64")
active_on = active_order[active_order["is_active"].astype(str) == "1"]
ao_group = active_on.groupby(["id_dealer_outlet","dealer_name"]).agg({"end_date":"min"}).reset_index().rename(columns={"end_date":"nearest_end_date"})

run_order = running_order.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
run_order["id_dealer_outlet"] = pd.to_numeric(run_order["id_dealer_outlet"], errors="coerce").astype("Int64")
run_order["active_dse"] = pd.to_numeric(run_order["active_dse"], errors="coerce").astype("Int64")
run_order = run_order.dropna(subset=["id_dealer_outlet"])
grouped_run_order = run_order.groupby(["id_dealer_outlet","dealer_name"]).agg({"joined_dse":"count","active_dse":"sum"}).reset_index()
grouped_run_order = grouped_run_order.merge(ao_group, how="left", on=["id_dealer_outlet","dealer_name"])

if not avail_df.empty:
    avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce").astype("Int64")

dist_cols = [c for c in avail_df.columns if c.startswith("dist_center_")]
if dist_cols:
    min_values = avail_df[dist_cols].apply(pd.to_numeric, errors="coerce").fillna(1e12).min(axis=1)
    for c in dist_cols:
        avail_df[c] = np.where(pd.to_numeric(avail_df[c], errors="coerce").round(6) == min_values.round(6), avail_df[c], np.nan)

avail_df_merge = avail_df.merge(grouped_run_order, how="left", on="id_dealer_outlet")
ld = location_detail.rename(columns={"City":"city","Cluster":"cluster"})
avail_df_merge = avail_df_merge.merge(ld[["city","cluster"]], how="left", on="city")
nc = cluster_left.replace({"CHERY":"Chery","Kia":"KIA"}).rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability","Category":"category"})
nc = nc[nc["category"].astype(str) == "Car"]
avail_df_merge = avail_df_merge.merge(nc[["cluster","brand","daily_gen","daily_need","delta","availability"]], how="left", on=["brand","cluster"])
avail_df_merge["tag"] = np.where(pd.isna(avail_df_merge["nearest_end_date"]), "Not Active", "Active")
