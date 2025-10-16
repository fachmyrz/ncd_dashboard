import pandas as pd
import numpy as np
import geopy.distance
from data_load import cluster_left, location_detail, df_visit, df_dealer, running_order
from sklearn.cluster import KMeans
from kneed import KneeLocator
from datetime import datetime
import re

_latlon_pat = re.compile(r"(-?\d+(?:\.\d+)?)")

def _to_float(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace("`", "").replace("’", "").replace("‘", "").strip()
    m = _latlon_pat.search(s)
    return float(m.group(1)) if m else np.nan

def _parse_latlon_field(s):
    if pd.isna(s):
        return np.nan, np.nan
    s = str(s).replace("`", "").replace("’", "").replace("‘", "").strip()
    parts = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(parts) >= 2:
        lat = float(parts[0])
        lon = float(parts[1])
        if abs(lat) > 90 and abs(lon) <= 90:
            lat, lon = lon, lat
        return lat, lon
    return np.nan, np.nan

def _is_valid_coord(lat, lon):
    return np.isfinite(lat) and np.isfinite(lon) and (-90 <= lat <= 90) and (-180 <= lon <= 180)

def clean_dealers(df):
    df = df.copy()
    if "business_type" in df.columns:
        df = df[df["business_type"].astype(str).str.strip().str.lower() == "car"]
    df["latitude"] = df["latitude"].apply(_to_float)
    df["longitude"] = df["longitude"].apply(_to_float)
    valid = df.apply(lambda r: _is_valid_coord(r["latitude"], r["longitude"]), axis=1)
    df = df[valid].reset_index(drop=True)
    df["name"] = df.get("name", "").astype(str).str.strip()
    df["city"] = df.get("city", "").astype(str).str.strip()
    return df

def clean_visits(df):
    df = df.copy()
    if "visit_datetime" in df.columns:
        df["visit_datetime"] = pd.to_datetime(df["visit_datetime"], errors="coerce")
    elif "Tanggal Datang" in df.columns:
        df["visit_datetime"] = pd.to_datetime(df["Tanggal Datang"], errors="coerce")
    elif "date_time_start" in df.columns:
        df["visit_datetime"] = pd.to_datetime(df["date_time_start"], errors="coerce")
    else:
        df["visit_datetime"] = pd.NaT
    if "Latitude & Longitude Datang" in df.columns:
        ll = df["Latitude & Longitude Datang"].apply(_parse_latlon_field)
        df["latitude"] = ll.apply(lambda t: t[0])
        df["longitude"] = ll.apply(lambda t: t[1])
    else:
        lat_col = None
        lon_col = None
        for c in df.columns:
            cs = c.lower()
            if lat_col is None and ("lat" in cs or "latitude" in cs):
                lat_col = c
            if lon_col is None and ("lon" in cs or "lng" in cs or "longitude" in cs):
                lon_col = c
        df["latitude"] = df[lat_col].apply(_to_float) if lat_col else np.nan
        df["longitude"] = df[lon_col].apply(_to_float) if lon_col else np.nan
    def _fix_swap(r):
        lat = r["latitude"]
        lon = r["longitude"]
        if np.isfinite(lat) and np.isfinite(lon) and (abs(lat) > 90) and (abs(lon) <= 90):
            return pd.Series({"latitude": lon, "longitude": lat})
        return pd.Series({"latitude": lat, "longitude": lon})
    swap = df.apply(_fix_swap, axis=1)
    df["latitude"] = swap["latitude"]
    df["longitude"] = swap["longitude"]
    valid = df.apply(lambda r: _is_valid_coord(r["latitude"], r["longitude"]), axis=1)
    df = df[valid].reset_index(drop=True)
    if "Nama Karyawan" in df.columns:
        df["employee_name"] = df["Nama Karyawan"].astype(str).str.strip()
    elif "employee_name" not in df.columns:
        df["employee_name"] = ""
    if "Nama Klien" in df.columns:
        df["client_name"] = df["Nama Klien"].astype(str).str.strip()
    elif "client_name" not in df.columns:
        df["client_name"] = ""
    if "Nomor Induk Karyawan" in df.columns:
        df["Nomor Induk Karyawan"] = df["Nomor Induk Karyawan"].astype(str)
    if "Divisi" in df.columns:
        df["Divisi"] = df["Divisi"].astype(str)
    return df

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    v = visits.copy()
    d = dealers.copy()
    d = d[(d["latitude"].apply(np.isfinite)) & (d["longitude"].apply(np.isfinite))]
    if d.empty or v.empty:
        v["matched_client"] = np.nan
        v["matched_dealer_id"] = np.nan
        return v
    d_idx = d[["id_dealer_outlet", "name", "latitude", "longitude"]].reset_index(drop=True)
    def match_row(row):
        lat = row["latitude"]
        lon = row["longitude"]
        name = str(row.get("client_name", "")).strip()
        if name:
            cand = d_idx[d_idx["name"].str.strip().str.lower() == name.lower()]
        else:
            cand = d_idx.iloc[0:0]
        if np.isfinite(lat) and np.isfinite(lon):
            if cand.empty:
                cand = d_idx
            dist = cand.apply(lambda x: geopy.distance.geodesic((lat, lon), (x["latitude"], x["longitude"])).km, axis=1)
            cand = cand.assign(_km=dist)
            near = cand[cand["_km"] <= max_km].sort_values("_km")
            if not near.empty:
                best = near.iloc[0]
                return pd.Series({"matched_client": best["name"], "matched_dealer_id": best["id_dealer_outlet"]})
        return pd.Series({"matched_client": np.nan, "matched_dealer_id": np.nan})
    m = v.apply(match_row, axis=1)
    v["matched_client"] = m["matched_client"]
    v["matched_dealer_id"] = m["matched_dealer_id"]
    return v

def get_summary_data(visits):
    summary = visits.copy()
    summary = summary[pd.notna(summary["visit_datetime"])]
    summary["date"] = pd.to_datetime(summary["visit_datetime"]).dt.date
    summary["lat"] = summary["latitude"].astype(float)
    summary["long"] = summary["longitude"].astype(float)
    summary.reset_index(drop=True, inplace=True)
    rows = []
    for dates in summary["date"].unique():
        for name in summary["employee_name"].dropna().unique():
            temp = summary[(summary.employee_name == name) & (summary["date"] == dates)].sort_values("visit_datetime").reset_index(drop=True)
            temp = temp[["date", "employee_name", "lat", "long", "visit_datetime"]]
            if len(temp) > 1:
                dist = []
                time_between = []
                for i in range(len(temp) - 1):
                    dkm = geopy.distance.geodesic((temp.loc[i + 1, "lat"], temp.loc[i + 1, "long"]), (temp.loc[i, "lat"], temp.loc[i, "long"])).km
                    dist.append(round(dkm, 2))
                    dtm = (pd.to_datetime(str(temp.loc[i + 1, "visit_datetime"])) - pd.to_datetime(str(temp.loc[i, "visit_datetime"]))).total_seconds() / 60
                    time_between.append(dtm)
                avg_speed = round(sum(dist) / sum(time_between), 2) if sum(time_between) != 0 else 0
                rows.append([dates, name, len(temp), round(np.mean(dist), 2), round(np.mean(time_between), 2), avg_speed])
            else:
                rows.append([dates, name, len(temp), 0, 0, 0])
    cols = ["date", "employee_name", "ctd_visit", "avg_distance_km", "avg_time_between_minute", "avg_speed_kmpm"]
    data = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    if not data.empty:
        data["date"] = pd.to_datetime(data["date"])
        data["month_year"] = data["date"].dt.strftime("%Y-%m")
    else:
        data["month_year"] = []
    return summary, data

df_dealer = clean_dealers(df_dealer)
df_visit = clean_visits(df_visit)
df_visit = df_visit[~df_visit.get("Nomor Induk Karyawan", pd.Series([], dtype=str)).astype(str).str.contains("deleted-", case=False, na=False)]
df_visit = df_visit[~df_visit.get("Divisi", pd.Series([], dtype=str)).astype(str).str.contains("trainer", case=False, na=False)]
df_visits = assign_visits_to_dealers(df_visit, df_dealer, max_km=1.0)
summary, data_sum = get_summary_data(df_visits)

filter_data = []
for name in summary.employee_name.dropna().unique():
    lat_long = summary[summary.employee_name == name][["lat", "long"]]
    if lat_long.empty:
        continue
    min_lat = lat_long["lat"].min()
    max_lat = lat_long["lat"].max()
    min_long = lat_long["long"].min()
    max_long = lat_long["long"].max()
    lat_ = geopy.distance.geodesic((max_lat, min_long), (min_lat, min_long)).km
    long_ = geopy.distance.geodesic((min_lat, max_long), (min_lat, min_long)).km
    area = lat_ * long_
    filter_data.append([name, min_lat, max_lat, min_long, max_long, area])
area_coverage = pd.DataFrame(data=filter_data, columns=["employee_name", "min_lat", "max_lat", "min_long", "max_long", "area"])
if not area_coverage.empty:
    area_coverage["min_lat"] = area_coverage["min_lat"].astype(float)
    area_coverage["min_long"] = area_coverage["min_long"].astype(float)
    area_coverage["max_lat"] = area_coverage["max_lat"].astype(float)
    area_coverage["max_long"] = area_coverage["max_long"].astype(float)

def get_distance_dealer(cluster, lat, long):
    return geopy.distance.geodesic((kmeans.cluster_centers_[cluster, 0], kmeans.cluster_centers_[cluster, 1]), (lat, long)).km

sum_data = []
avail_data = []
cluster_center = []

for name in area_coverage.employee_name.unique():
    data_ = area_coverage[area_coverage.employee_name == name]
    get_dealer = df_dealer[(df_dealer.latitude.between(data_.min_lat.values[0], data_.max_lat.values[0])) & (df_dealer.longitude.between(data_.min_long.values[0], data_.max_long.values[0]))]
    sum_ = summary[summary.employee_name == name][["date", "client_name", "lat", "long"]].rename(columns={"lat": "latitude", "long": "longitude"}).copy()
    sum_["sales_name"] = name
    avail_ = get_dealer[["id_dealer_outlet", "brand", "business_type", "city", "name", "latitude", "longitude"]].copy()
    avail_["tag"] = "avail"
    avail_["sales_name"] = name
    if len(sum_) >= 2:
        wcss = []
        ub = min(9, max(5, len(sum_)))
        for i in range(4, ub):
            X = list(zip(sum_["latitude"], sum_["longitude"]))
            km = KMeans(n_clusters=i, n_init=10).fit(X)
            wcss.append(km.inertia_)
        knee = KneeLocator(range(4, ub), wcss, curve="convex", direction="decreasing")
        n_cluster = knee.elbow if getattr(knee, "elbow", None) else 4
        kmeans = KMeans(n_clusters=n_cluster, n_init=10)
        data_ll = list(zip(sum_["latitude"], sum_["longitude"]))
        kmeans.fit(data_ll)
        sum_["cluster"] = kmeans.labels_
        for i in range(len(kmeans.cluster_centers_)):
            avail_[f"dist_center_{i}"] = avail_.apply(lambda x: get_distance_dealer(i, x.latitude, x.longitude), axis=1)
    else:
        kmeans = KMeans(n_clusters=1, n_init=10)
        data_ll = list(zip(sum_["latitude"], sum_["longitude"])) if not sum_.empty else [(get_dealer["latitude"].mean(), get_dealer["longitude"].mean())]
        kmeans.fit(data_ll)
        sum_["cluster"] = 0
        for i in range(len(kmeans.cluster_centers_)):
            avail_[f"dist_center_{i}"] = avail_.apply(lambda x: get_distance_dealer(i, x.latitude, x.longitude), axis=1)
    clust_ = pd.DataFrame(kmeans.cluster_centers_, columns=["latitude", "longitude"])
    clust_["sales_name"] = name
    clust_["cluster"] = range(len(kmeans.cluster_centers_))
    cluster_center.append(clust_)
    avail_data.append(avail_)
    sum_data.append(sum_)

sum_df = pd.concat(sum_data) if sum_data else pd.DataFrame(columns=["date", "client_name", "latitude", "longitude", "sales_name", "cluster"])
avail_df = pd.concat(avail_data) if avail_data else pd.DataFrame(columns=["id_dealer_outlet", "brand", "business_type", "city", "name", "latitude", "longitude", "tag", "sales_name"])
clust_df = pd.concat(cluster_center) if cluster_center else pd.DataFrame(columns=["latitude", "longitude", "sales_name", "cluster"])

active_order = running_order.rename(columns={"Dealer Id": "id_dealer_outlet", "Dealer Name": "dealer_name", "IsActive": "IsActive", "End Date": "End Date"})
active_order["End Date"] = pd.to_datetime(active_order.get("End Date"), errors="coerce")
active_order["id_dealer_outlet"] = pd.to_numeric(active_order.get("id_dealer_outlet"), errors="coerce")
ao = active_order[active_order.get("IsActive").astype(str) == "1"].copy()
ao_group = ao.groupby(["id_dealer_outlet", "dealer_name"]).agg({"End Date": "min"}).reset_index().rename(columns={"End Date": "nearest_end_date"})

run_order = running_order.rename(columns={"Dealer Id": "id_dealer_outlet", "Dealer Name": "dealer_name", "LMS Id": "joined_dse", "IsActive": "active_dse"}).copy()
run_order["id_dealer_outlet"] = pd.to_numeric(run_order.get("id_dealer_outlet"), errors="coerce")
run_order["active_dse"] = pd.to_numeric(run_order.get("active_dse"), errors="coerce")
run_order = run_order[~run_order["id_dealer_outlet"].isna()]
grouped_run_order = run_order.groupby(["id_dealer_outlet", "dealer_name"]).agg({"joined_dse": "count", "active_dse": "sum"}).reset_index()
grouped_run_order = grouped_run_order.merge(ao_group, how="left", on=["id_dealer_outlet", "dealer_name"])
avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce")

if not avail_df.empty:
    dist_cols = [c for c in avail_df.columns if str(c).startswith("dist_center_")]
    if dist_cols:
        min_values = avail_df.fillna(1e12)[dist_cols].min(axis=1)
        avail_df[dist_cols] = avail_df[dist_cols].where(avail_df[dist_cols].eq(min_values, axis=0), np.nan)

avail_df_merge = avail_df.merge(grouped_run_order, how="left", on="id_dealer_outlet")
ld = location_detail.rename(columns={"City": "city", "Cluster": "cluster"})
avail_df_merge = avail_df_merge.merge(ld[["city", "cluster"]], how="left", on="city")
cl_map = cluster_left.replace({"Brand": {"CHERY": "Chery", "Kia": "KIA"}})
cl_map = cl_map[cl_map.get("Category", "").astype(str) == "Car"].rename(columns={"Cluster": "cluster", "Brand": "brand", "Daily_Gen": "daily_gen", "Daily_Need": "daily_need", "Delta": "delta", "Tag": "availability"})
avail_df_merge = avail_df_merge.merge(cl_map[["cluster", "brand", "daily_gen", "daily_need", "delta", "availability"]], how="left", on=["brand", "cluster"])
avail_df_merge["tag"] = np.where(pd.to_datetime(avail_df_merge.get("nearest_end_date"), errors="coerce").notna(), "Active", "Not Active")

visit_stats = df_visits.dropna(subset=["matched_dealer_id"]).copy()
visit_stats["matched_dealer_id"] = pd.to_numeric(visit_stats["matched_dealer_id"], errors="coerce")
wk = visit_stats.groupby(["matched_dealer_id"]).agg(
    last_visit=("visit_datetime", "max"),
    last_bde=("employee_name", "last"),
    visits=("matched_dealer_id", "count")
).reset_index()
if not visit_stats.empty:
    visit_stats["week"] = pd.to_datetime(visit_stats["visit_datetime"]).dt.isocalendar().week
    weekly = visit_stats.groupby(["matched_dealer_id", "week"]).size().reset_index(name="wvis")
    avg_week = weekly.groupby("matched_dealer_id")["wvis"].mean().reset_index().rename(columns={"wvis": "avg_weekly_visits"})
else:
    avg_week = pd.DataFrame(columns=["matched_dealer_id", "avg_weekly_visits"])
wk = wk.merge(avg_week, how="left", on="matched_dealer_id")
wk["avg_weekly_visits"] = wk["avg_weekly_visits"].fillna(0)
avail_df_merge = avail_df_merge.merge(wk.rename(columns={"matched_dealer_id": "id_dealer_outlet"}), how="left", on="id_dealer_outlet")
avail_df_merge["avg_weekly_visits"] = avail_df_merge["avg_weekly_visits"].fillna(0)
avail_df_merge["last_visit"] = pd.to_datetime(avail_df_merge["last_visit"], errors="coerce")
avail_df_merge["last_bde"] = avail_df_merge["last_bde"].astype(str).fillna("")
