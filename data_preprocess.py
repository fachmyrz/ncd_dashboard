import re
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import data_load

def _first_col(df, candidates, default=None):
    if df is None or df.empty:
        return default
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return default

def _to_float_safe(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = s.strip()
    s = s.replace('`', '').replace(',', '')
    s = re.sub(r'[^\d\.\-]', '', s)
    try:
        return float(s)
    except:
        return np.nan

def haversine_np(lat1, lon1, lats2, lons2):
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lats2 = np.asarray(lats2, dtype=float)
    lons2 = np.asarray(lons2, dtype=float)
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lats2)
    lon2r = np.radians(lons2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0088 * c
    return km

def clean_dealers(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    id_col = _first_col(df, ["id_dealer_outlet", "Dealer Id", "id_dealer", "dealer_id"], default=df.columns[0])
    brand_col = _first_col(df, ["brand", "Brand"])
    bt_col = _first_col(df, ["business_type", "Business Type", "type"])
    city_col = _first_col(df, ["city", "City", "kota"])
    name_col = _first_col(df, ["name", "Name", "Dealer Name", "dealer_name"])
    lat_col = _first_col(df, ["latitude", "Latitude", "lat"])
    lon_col = _first_col(df, ["longitude", "Longitude", "long", "lon"])
    cols_map = {}
    if id_col: cols_map[id_col] = "id_dealer_outlet"
    if brand_col: cols_map[brand_col] = "brand"
    if bt_col: cols_map[bt_col] = "business_type"
    if city_col: cols_map[city_col] = "city"
    if name_col: cols_map[name_col] = "name"
    if lat_col: cols_map[lat_col] = "latitude"
    if lon_col: cols_map[lon_col] = "longitude"
    df = df.rename(columns=cols_map)
    if "business_type" in df.columns:
        df = df[df["business_type"].astype(str).str.lower().str.contains("car", na=False)]
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].apply(_to_float_safe)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].apply(_to_float_safe)
    if "id_dealer_outlet" in df.columns:
        try:
            df["id_dealer_outlet"] = df["id_dealer_outlet"].astype(int)
        except:
            df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce").fillna(-1).astype(int)
    df = df.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    return df

def _parse_latlon_field(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    try:
        s = str(s).strip()
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 2:
                lat = _to_float_safe(parts[0])
                lon = _to_float_safe(parts[1])
                return (lat, lon)
        nums = re.findall(r"-?\d+\.\d+|-?\d+", s)
        if len(nums) >= 2:
            return (_to_float_safe(nums[0]), _to_float_safe(nums[1]))
    except:
        pass
    return (np.nan, np.nan)

def clean_visits(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    col_employee = _first_col(df, ["Nama Karyawan","Employee Name","employee_name","Nama Karyawan"])
    col_client = _first_col(df, ["Nama Klien","Client Name","client_name","Client"])
    col_date = _first_col(df, ["Tanggal Datang","Date Time Start","date_time_start","Order Date","Tanggal"])
    col_latlon = _first_col(df, ["Latitude & Longitude Datang","Latitude & Longitude","latitude_longitude","latlong","Latitude & Longitude"])
    col_nik = _first_col(df, ["Nomor Induk Karyawan","Nomor Induk","NIK","nik"])
    col_divisi = _first_col(df, ["Divisi","Division","divisi"])
    remap = {}
    if col_employee: remap[col_employee] = "employee_name"
    if col_client: remap[col_client] = "client_name"
    if col_date: remap[col_date] = "visit_datetime"
    if col_latlon: remap[col_latlon] = "latlong"
    if col_nik: remap[col_nik] = "nik"
    if col_divisi: remap[col_divisi] = "divisi"
    df = df.rename(columns=remap)
    if "visit_datetime" in df.columns:
        df["visit_datetime"] = pd.to_datetime(df["visit_datetime"].astype(str).str.strip(), errors="coerce")
    if "latlong" in df.columns:
        parsed = df["latlong"].apply(_parse_latlon_field)
        df["lat"] = parsed.apply(lambda t: t[0])
        df["long"] = parsed.apply(lambda t: t[1])
    else:
        latc = _first_col(df, ["latitude","Latitude","lat"])
        lonc = _first_col(df, ["longitude","Longitude","long"])
        if latc and lonc:
            df["lat"] = df[latc].apply(_to_float_safe)
            df["long"] = df[lonc].apply(_to_float_safe)
    if "nik" in df.columns:
        df["nik"] = df["nik"].astype(str)
        df = df[~df["nik"].str.contains("deleted-", na=False)]
    if "divisi" in df.columns:
        df = df[~df["divisi"].astype(str).str.contains("trainer", case=False, na=False)]
    if "employee_name" in df.columns:
        df["employee_name"] = df["employee_name"].astype(str).str.strip()
    df = df.dropna(subset=["employee_name"]).reset_index(drop=True)
    return df

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    if visits is None or dealers is None:
        return pd.DataFrame()
    v = visits.copy()
    d = dealers.copy()
    if "id_dealer_outlet" not in d.columns:
        d = d.reset_index().rename(columns={"index":"id_dealer_outlet"})
    dcoords = d[["id_dealer_outlet","name","latitude","longitude"]].dropna().reset_index(drop=True)
    d_lat = dcoords["latitude"].to_numpy(dtype=float)
    d_lon = dcoords["longitude"].to_numpy(dtype=float)
    def _match_row(row):
        lat = _to_float_safe(row.get("lat", np.nan))
        lon = _to_float_safe(row.get("long", np.nan))
        cname = row.get("client_name", "")
        if pd.notna(cname) and cname != "" :
            cand = dcoords[dcoords["name"].astype(str).str.lower() == str(cname).strip().lower()]
            if len(cand) == 1:
                return cand.iloc[0]["name"], int(cand.iloc[0]["id_dealer_outlet"])
        if pd.isna(lat) or pd.isna(lon):
            return (np.nan, np.nan)
        lat_arr = np.full_like(d_lat, lat, dtype=float)
        dist = haversine_np(lat_arr, lon, d_lat, d_lon)
        idx = np.nanargmin(dist)
        km = float(dist[idx]) if len(dist)>0 else np.nan
        if km <= max_km:
            return (dcoords.iloc[idx]["name"], int(dcoords.iloc[idx]["id_dealer_outlet"]))
        return (np.nan, np.nan)
    matched = v.apply(lambda r: _match_row(r), axis=1)
    matched_df = pd.DataFrame(matched.tolist(), columns=["matched_name","matched_dealer_id"])
    v = pd.concat([v.reset_index(drop=True), matched_df], axis=1)
    return v

def prepare_run_order(running_order):
    if running_order is None or running_order.empty:
        return pd.DataFrame()
    df = running_order.copy()
    id_col = _first_col(df, ["Dealer Id","id_dealer_outlet","Dealer Id"])
    dealer_name_col = _first_col(df, ["Dealer Name","dealer_name","Dealer"])
    joined_col = _first_col(df, ["LMS Id","joined_dse","LMS Id"])
    active_col = _first_col(df, ["IsActive","active_dse","Is Active"])
    enddate_col = _first_col(df, ["End Date","EndDate","nearest_end_date"])
    rem = {}
    if id_col: rem[id_col] = "id_dealer_outlet"
    if dealer_name_col: rem[dealer_name_col] = "dealer_name"
    if joined_col: rem[joined_col] = "joined_dse"
    if active_col: rem[active_col] = "active_dse"
    if enddate_col: rem[enddate_col] = "nearest_end_date"
    df = df.rename(columns=rem)
    if "id_dealer_outlet" in df.columns:
        df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce")
    df["joined_dse"] = pd.to_numeric(df.get("joined_dse", pd.Series(0)), errors="coerce").fillna(0).astype(int)
    df["active_dse"] = pd.to_numeric(df.get("active_dse", pd.Series(0)), errors="coerce").fillna(0).astype(int)
    if "nearest_end_date" in df.columns:
        df["nearest_end_date"] = pd.to_datetime(df["nearest_end_date"], errors="coerce")
    grouped = df.groupby("id_dealer_outlet", dropna=True).agg({"joined_dse":"sum","active_dse":"sum","nearest_end_date":"min"}).reset_index()
    return grouped

def compute_clusters(visits):
    if visits is None or visits.empty:
        return pd.DataFrame(), pd.DataFrame()
    visits = visits.dropna(subset=["lat","long"])
    res_centers = []
    sum_rows = []
    for name in visits["employee_name"].dropna().unique():
        subset = visits[visits["employee_name"] == name]
        pts = subset[["lat","long"]].dropna()
        if pts.empty:
            continue
        n = min(4, max(1, int(len(pts)/5))) if len(pts) >= 2 else 1
        if n < 1:
            n = 1
        if len(pts) < n:
            n = max(1, len(pts))
        try:
            k = KMeans(n_clusters=n, random_state=42).fit(pts.to_numpy())
            labels = k.labels_
            centers = k.cluster_centers_
        except:
            labels = np.zeros(len(pts), dtype=int)
            centers = np.array([pts.mean().tolist()])
        pts = pts.reset_index(drop=True)
        pts["cluster"] = labels
        pts["sales_name"] = name
        for i, c in enumerate(centers):
            res_centers.append({"sales_name":name,"cluster":i,"latitude":float(c[0]),"longitude":float(c[1])})
        sum_rows.append(pts.assign(employee_name=name))
    clust_df = pd.DataFrame(res_centers)
    sum_df = pd.concat(sum_rows, ignore_index=True) if sum_rows else pd.DataFrame()
    return sum_df, clust_df

def compute_availability(dealers, run_order_group, location_detail, cluster_centers):
    avail = dealers.copy()
    if "id_dealer_outlet" in avail.columns:
        avail["id_dealer_outlet"] = pd.to_numeric(avail["id_dealer_outlet"], errors="coerce").astype("Int64")
    dist_cols = []
    if cluster_centers is None or cluster_centers.empty:
        cluster_centers = pd.DataFrame()
    else:
        for i in cluster_centers["cluster"].unique():
            row = cluster_centers[cluster_centers["cluster"]==i].iloc[0]
            lat_c = float(row["latitude"])
            lon_c = float(row["longitude"])
            col = f"dist_center_{i}"
            avail[col] = haversine_np(avail["latitude"].to_numpy(dtype=float), avail["longitude"].to_numpy(dtype=float), np.full(len(avail), lat_c), np.full(len(avail), lon_c))
            dist_cols.append(col)
    if run_order_group is not None and not run_order_group.empty:
        cols_to_merge = [c for c in run_order_group.columns if c!="dealer_name"]
        try:
            avail = avail.merge(run_order_group[cols_to_merge], how="left", on="id_dealer_outlet")
        except:
            avail = avail.merge(run_order_group, how="left", left_on="id_dealer_outlet", right_on=run_order_group.columns[0])
    if location_detail is not None and not location_detail.empty:
        ld = location_detail.copy()
        map_cols = {}
        ccity = _first_col(ld, ["City","city","City Name"])
        ccluster = _first_col(ld, ["Cluster","cluster","area"])
        if ccity and ccluster:
            ld = ld.rename(columns={ccity:"city", ccluster:"cluster"})
            try:
                avail = avail.merge(ld[["city","cluster"]], how="left", on="city")
            except:
                pass
    return avail

def get_summary_data(visits):
    if visits is None or visits.empty:
        return pd.DataFrame(), pd.DataFrame()
    v = visits.copy()
    if "visit_datetime" in v.columns:
        v["date"] = pd.to_datetime(v["visit_datetime"], errors="coerce").dt.date
    else:
        v["date"] = pd.NaT
    v = v.dropna(subset=["date"])
    v["month_year"] = pd.to_datetime(v["date"]).to_period("M").astype(str)
    data = []
    for (m, name), grp in v.groupby(["month_year","employee_name"]):
        coords = grp[["lat","long"]].dropna()
        if not coords.empty and len(coords) > 1:
            dists = []
            times = []
            lat_arr = coords["lat"].to_numpy()
            lon_arr = coords["long"].to_numpy()
            for i in range(len(lat_arr)-1):
                d = haversine_np(lat_arr[i], lon_arr[i], np.array([lat_arr[i+1]]), np.array([lon_arr[i+1]]))[0]
                dists.append(d)
            avg_dist = float(np.mean(dists)) if dists else 0.0
        else:
            avg_dist = 0.0
        data.append({"month_year":m,"employee_name":name,"ctd_visit":len(grp),"avg_distance_km":avg_dist,"avg_time_between_minute":0.0,"avg_speed_kmpm":0.0})
    df = pd.DataFrame(data)
    return df, df

def compute_all():
    try:
        dealers = data_load.df_dealer if hasattr(data_load, "df_dealer") else pd.DataFrame()
        visits = data_load.df_visit if hasattr(data_load, "df_visit") else pd.DataFrame()
        location_detail = data_load.location_detail if hasattr(data_load, "location_detail") else pd.DataFrame()
        need_cluster = data_load.cluster_left if hasattr(data_load, "cluster_left") else pd.DataFrame()
        running_order = data_load.running_order if hasattr(data_load, "running_order") else pd.DataFrame()
    except:
        dealers = pd.DataFrame()
        visits = pd.DataFrame()
        location_detail = pd.DataFrame()
        need_cluster = pd.DataFrame()
        running_order = pd.DataFrame()
    dealers = clean_dealers(dealers)
    visits = clean_visits(visits)
    visits = visits.reset_index(drop=True)
    visits = assign_visits_to_dealers(visits, dealers, max_km=1.0)
    sum_df, clust_df = compute_clusters(visits)
    ro_group = prepare_run_order(running_order)
    avail_df_merge = compute_availability(dealers, ro_group, location_detail, clust_df)
    return {"sum_df":sum_df,"clust_df":clust_df,"avail_df_merge":avail_df_merge,"df_visits":visits}
