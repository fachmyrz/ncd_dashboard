import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import geopy.distance
import re
from datetime import datetime
import streamlit as st
from data_load import cluster_left, location_detail, df_visit, df_dealer, running_order

pat_num = re.compile(r"-?\d+(?:\.\d+)?")

def _num(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace("`", "").replace("’", "").replace("‘", "").strip()
    m = pat_num.search(s)
    return float(m.group(0)) if m else np.nan

def _pair(s):
    if pd.isna(s):
        return np.nan, np.nan
    s = str(s).replace("`", "").replace("’", "").replace("‘", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) >= 2:
        a = float(nums[0])
        b = float(nums[1])
        if abs(a) > 90 and abs(b) <= 90:
            a, b = b, a
        return a, b
    return np.nan, np.nan

def _ok(lat, lon):
    return np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180

@st.cache_data(ttl=300, show_spinner=False)
def load_sources():
    d = df_dealer.copy()
    if "business_type" in d.columns:
        d = d[d["business_type"].astype(str).str.strip().str.lower() == "car"]
    d["latitude"] = d["latitude"].apply(_num)
    d["longitude"] = d["longitude"].apply(_num)
    d = d[d.apply(lambda r: _ok(r["latitude"], r["longitude"]), axis=1)].reset_index(drop=True)
    d["name"] = d.get("name", "").astype(str).str.strip()
    d["city"] = d.get("city", "").astype(str).str.strip()
    v = df_visit.copy()
    if "visit_datetime" in v.columns:
        v["visit_datetime"] = pd.to_datetime(v["visit_datetime"], errors="coerce")
    elif "Tanggal Datang" in v.columns:
        v["visit_datetime"] = pd.to_datetime(v["Tanggal Datang"], errors="coerce")
    elif "Date Time Start" in v.columns:
        v["visit_datetime"] = pd.to_datetime(v["Date Time Start"].astype(str).str.replace("@"," "), errors="coerce")
    else:
        v["visit_datetime"] = pd.NaT
    if "Latitude & Longitude Datang" in v.columns:
        ll = v["Latitude & Longitude Datang"].apply(_pair)
        v["latitude"] = ll.apply(lambda t: t[0])
        v["longitude"] = ll.apply(lambda t: t[1])
    else:
        latc = None
        lonc = None
        for c in v.columns:
            cl = c.lower()
            if latc is None and ("lat" in cl or "latitude" in cl):
                latc = c
            if lonc is None and ("lon" in cl or "lng" in cl or "longitude" in cl):
                lonc = c
        v["latitude"] = v[latc].apply(_num) if latc else np.nan
        v["longitude"] = v[lonc].apply(_num) if lonc else np.nan
    v = v[v.apply(lambda r: _ok(r["latitude"], r["longitude"]), axis=1)].reset_index(drop=True)
    if "Nama Karyawan" in v.columns:
        v["employee_name"] = v["Nama Karyawan"].astype(str).str.strip()
    elif "employee_name" not in v.columns:
        v["employee_name"] = ""
    if "Nama Klien" in v.columns:
        v["client_name"] = v["Nama Klien"].astype(str).str.strip()
    elif "client_name" not in v.columns:
        v["client_name"] = ""
    if "Nomor Induk Karyawan" in v.columns:
        v["Nomor Induk Karyawan"] = v["Nomor Induk Karyawan"].astype(str)
    if "Divisi" in v.columns:
        v["Divisi"] = v["Divisi"].astype(str)
    ro = running_order.copy()
    ld = location_detail.rename(columns={"City": "city", "Cluster": "cluster"}).copy()
    nc = cluster_left.replace({"Brand": {"CHERY": "Chery", "Kia": "KIA"}}).copy()
    return d, v, ro, ld, nc

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    v = visits.copy()
    d = dealers.copy()
    if v.empty or d.empty:
        v["matched_dealer_id"] = np.nan
        v["matched_client"] = np.nan
        return v
    left = v.merge(d[["id_dealer_outlet", "name"]], left_on="client_name", right_on="name", how="left")
    v["matched_dealer_id"] = left["id_dealer_outlet"]
    v["matched_client"] = left["name"]
    mask_un = v["matched_dealer_id"].isna()
    if mask_un.any():
        dv = d[["id_dealer_outlet", "latitude", "longitude"]].dropna().reset_index(drop=True)
        if not dv.empty:
            xv = np.radians(v.loc[mask_un, ["latitude", "longitude"]].values.astype(float))
            xd = np.radians(dv[["latitude", "longitude"]].values.astype(float))
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric="haversine").fit(xd)
            dist, idx = nbrs.kneighbors(xv)
            dist_km = dist[:, 0] * 6371.0088
            near = dist_km <= max_km
            ids = np.full(mask_un.sum(), np.nan)
            names = np.full(mask_un.sum(), np.nan, dtype=object)
            sel = idx[near, 0].reshape(-1)
            if sel.size > 0:
                ids[near] = dv.iloc[sel]["id_dealer_outlet"].values
                names[near] = dv.iloc[sel]["id_dealer_outlet"].astype(str).values
            v.loc[mask_un, "matched_dealer_id"] = ids
            v.loc[mask_un, "matched_client"] = names
    return v

def get_summary_data(visits):
    s = visits.copy()
    s = s[pd.notna(s["visit_datetime"])]
    s["date"] = pd.to_datetime(s["visit_datetime"]).dt.date
    s["lat"] = s["latitude"].astype(float)
    s["long"] = s["longitude"].astype(float)
    s.reset_index(drop=True, inplace=True)
    rows = []
    for dates in s["date"].unique():
        for name in s["employee_name"].dropna().unique():
            t = s[(s.employee_name == name) & (s["date"] == dates)].sort_values("visit_datetime").reset_index(drop=True)
            t = t[["date", "employee_name", "lat", "long", "visit_datetime"]]
            if len(t) > 1:
                dist = []
                mins = []
                for i in range(len(t) - 1):
                    dkm = geopy.distance.geodesic((t.loc[i + 1, "lat"], t.loc[i + 1, "long"]), (t.loc[i, "lat"], t.loc[i, "long"])).km
                    dist.append(round(dkm, 2))
                    dm = (pd.to_datetime(str(t.loc[i + 1, "visit_datetime"])) - pd.to_datetime(str(t.loc[i, "visit_datetime"]))).total_seconds() / 60
                    mins.append(dm)
                spd = round(sum(dist) / sum(mins), 2) if sum(mins) != 0 else 0
                rows.append([dates, name, len(t), round(np.mean(dist), 2), round(np.mean(mins), 2), spd])
            else:
                rows.append([dates, name, len(t), 0, 0, 0])
    cols = ["date", "employee_name", "ctd_visit", "avg_distance_km", "avg_time_between_minute", "avg_speed_kmpm"]
    data = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    if not data.empty:
        data["date"] = pd.to_datetime(data["date"])
        data["month_year"] = data["date"].dt.strftime("%Y-%m")
    else:
        data["month_year"] = []
    return s, data

def cluster_for_name(name, area_coverage, df_dealer, summary):
    d_ = area_coverage[area_coverage.employee_name == name]
    if d_.empty:
        return pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"]), pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","latitude","longitude","tag","sales_name"]), pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])
    get_dealer = df_dealer[(df_dealer.latitude.between(d_.min_lat.values[0], d_.max_lat.values[0])) & (df_dealer.longitude.between(d_.min_long.values[0], d_.max_long.values[0]))]
    s = summary[summary.employee_name == name][["date", "client_name", "lat", "long"]].rename(columns={"lat": "latitude", "long": "longitude"}).copy()
    s["sales_name"] = name
    a = get_dealer[["id_dealer_outlet", "brand", "business_type", "city", "name", "latitude", "longitude"]].copy()
    a["tag"] = "avail"
    a["sales_name"] = name
    if len(s) < 2:
        centers = np.array([[get_dealer["latitude"].mean(), get_dealer["longitude"].mean()]])
        s["cluster"] = 0
    else:
        k = min(4, max(1, len(s)))
        km = KMeans(n_clusters=k, n_init=10).fit(list(zip(s["latitude"], s["longitude"])))
        s["cluster"] = km.labels_
        centers = km.cluster_centers_
    for i in range(len(centers)):
        a[f"dist_center_{i}"] = a.apply(lambda r: geopy.distance.geodesic((centers[i, 0], centers[i, 1]), (r.latitude, r.longitude)).km, axis=1)
    c = pd.DataFrame(centers, columns=["latitude", "longitude"])
    c["sales_name"] = name
    c["cluster"] = range(len(centers))
    return s, a, c

def compute_all(bde_name, radius_km, area_pick, city_pick, brand_pick, penetrated, potential):
    dealers, visits, ro, ld, nc = load_sources()
    visits = visits[~visits.get("Nomor Induk Karyawan", pd.Series([], dtype=str)).astype(str).str.contains("deleted-", case=False, na=False)]
    visits = visits[~visits.get("Divisi", pd.Series([], dtype=str)).astype(str).str.contains("trainer", case=False, na=False)]
    vmatch = assign_visits_to_dealers(visits, dealers, max_km=1.0)
    summary, _ = get_summary_data(vmatch)
    filt = []
    for n in summary.employee_name.dropna().unique():
        points = summary[summary.employee_name == n][["lat", "long"]]
        if points.empty:
            continue
        min_lat = points["lat"].min()
        max_lat = points["lat"].max()
        min_lon = points["long"].min()
        max_lon = points["long"].max()
        lat_km = geopy.distance.geodesic((max_lat, min_lon), (min_lat, min_lon)).km
        lon_km = geopy.distance.geodesic((min_lat, max_lon), (min_lat, min_lon)).km
        area = lat_km * lon_km
        filt.append([n, min_lat, max_lat, min_lon, max_lon, area])
    ac = pd.DataFrame(filt, columns=["employee_name", "min_lat", "max_lat", "min_long", "max_long", "area"])
    if ac.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    if bde_name == "All":
        sum_list = []
        avail_list = []
        cl_list = []
        for n in ac.employee_name.unique():
            s, a, c = cluster_for_name(n, ac, dealers, summary)
            sum_list.append(s)
            avail_list.append(a)
            cl_list.append(c)
        sum_df = pd.concat(sum_list, ignore_index=True) if sum_list else pd.DataFrame()
        avail_df = pd.concat(avail_list, ignore_index=True) if avail_list else pd.DataFrame()
        clust_df = pd.concat(cl_list, ignore_index=True) if cl_list else pd.DataFrame()
    else:
        s, a, c = cluster_for_name(bde_name, ac, dealers, summary)
        sum_df = s
        avail_df = a
        clust_df = c
    run = ro.rename(columns={"Dealer Id": "id_dealer_outlet", "Dealer Name": "dealer_name", "LMS Id": "joined_dse", "IsActive": "active_dse", "End Date": "End Date"}).copy()
    run["id_dealer_outlet"] = pd.to_numeric(run["id_dealer_outlet"], errors="coerce")
    run["active_dse"] = pd.to_numeric(run["active_dse"], errors="coerce")
    run = run[~run["id_dealer_outlet"].isna()]
    active = run[run["active_dse"].astype(str) == "1"][["id_dealer_outlet", "dealer_name", "End Date"]].copy()
    active["End Date"] = pd.to_datetime(active["End Date"], errors="coerce")
    ao = active.groupby(["id_dealer_outlet", "dealer_name"]).agg({"End Date": "min"}).reset_index().rename(columns={"End Date": "nearest_end_date"})
    grouped = run.groupby(["id_dealer_outlet", "dealer_name"]).agg({"joined_dse": "count", "active_dse": "sum"}).reset_index()
    grouped = grouped.merge(ao, how="left", on=["id_dealer_outlet", "dealer_name"])
    avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce")
    dist_cols = [c for c in avail_df.columns if str(c).startswith("dist_center_")]
    if dist_cols:
        mv = avail_df.fillna(1e12)[dist_cols].min(axis=1)
        avail_df[dist_cols] = avail_df[dist_cols].where(avail_df[dist_cols].eq(mv, axis=0), np.nan)
    avail_df_merge = avail_df.merge(grouped.drop(columns=["dealer_name"]), how="left", on="id_dealer_outlet")
    ld_map = location_detail.rename(columns={"City":"city","Cluster":"cluster"})[["city","cluster"]]
    avail_df_merge = avail_df_merge.merge(ld_map, how="left", on="city")
    nc_map = cluster_left.replace({"Brand":{"CHERY":"Chery","Kia":"KIA"}})
    nc_map = nc_map[nc_map.get("Category","").astype(str) == "Car"].rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"})
    avail_df_merge = avail_df_merge.merge(nc_map[["cluster","brand","daily_gen","daily_need","delta","availability"]], how="left", on=["brand","cluster"])
    avail_df_merge["tag"] = np.where(pd.to_datetime(avail_df_merge.get("nearest_end_date"), errors="coerce").notna(), "Active", "Not Active")
    vstats = vmatch.dropna(subset=["matched_dealer_id"]).copy()
    vstats["matched_dealer_id"] = pd.to_numeric(vstats["matched_dealer_id"], errors="coerce")
    wk = vstats.groupby(["matched_dealer_id"]).agg(last_visit=("visit_datetime", "max"), last_bde=("employee_name", "last"), visits=("matched_dealer_id", "count")).reset_index()
    if not vstats.empty:
        vstats["week"] = pd.to_datetime(vstats["visit_datetime"]).dt.isocalendar().week
        w = vstats.groupby(["matched_dealer_id", "week"]).size().reset_index(name="wvis")
        avgw = w.groupby("matched_dealer_id")["wvis"].mean().reset_index().rename(columns={"wvis": "avg_weekly_visits"})
    else:
        avgw = pd.DataFrame(columns=["matched_dealer_id", "avg_weekly_visits"])
    wk = wk.merge(avgw, how="left", on="matched_dealer_id")
    wk["avg_weekly_visits"] = wk["avg_weekly_visits"].fillna(0)
    avail_df_merge = avail_df_merge.merge(wk.rename(columns={"matched_dealer_id": "id_dealer_outlet"}), how="left", on="id_dealer_outlet")
    avail_df_merge["avg_weekly_visits"] = avail_df_merge["avg_weekly_visits"].fillna(0)
    avail_df_merge["last_visit"] = pd.to_datetime(avail_df_merge["last_visit"], errors="coerce")
    avail_df_merge["last_bde"] = avail_df_merge["last_bde"].astype(str).fillna("")
    if area_pick:
        avail_df_merge = avail_df_merge[avail_df_merge["cluster"].astype(str).isin([str(x) for x in area_pick])]
    if city_pick:
        avail_df_merge = avail_df_merge[avail_df_merge["city"].astype(str).isin(city_pick)]
    if brand_pick:
        avail_df_merge = avail_df_merge[avail_df_merge["brand"].astype(str).isin(brand_pick)]
    if penetrated:
        avail_df_merge = avail_df_merge[avail_df_merge["tag"].astype(str).isin(penetrated)]
    if potential:
        avail_df_merge = avail_df_merge[avail_df_merge["availability"].astype(str).isin(potential)]
    picks = []
    for i in range(len([c for c in avail_df_merge.columns if str(c).startswith("dist_center_")]) or 1):
        col = f"dist_center_{i}"
        if col in avail_df_merge.columns:
            t = avail_df_merge[(avail_df_merge[col].notna()) & (avail_df_merge[col] <= radius_km)]
            if not t.empty:
                k = t.copy()
                k["cluster_labels"] = i
                picks.append(k)
    pick = pd.concat(picks, ignore_index=True) if picks else avail_df_merge.iloc[0:0].copy()
    centers = clust_df = None
    if bde_name == "All":
        tmp = []
        for n in ac.employee_name.unique():
            s, a, c = cluster_for_name(n, ac, dealers, summary)
            if not c.empty:
                tmp.append(c)
        clust_df = pd.concat(tmp, ignore_index=True) if tmp else pd.DataFrame(columns=["latitude","longitude","cluster"])
        if not clust_df.empty:
            centers = clust_df.groupby("cluster", as_index=False).agg({"latitude":"mean","longitude":"mean"})
            centers["sales_name"] = "All"
        else:
            centers = pd.DataFrame(columns=["latitude","longitude","cluster","sales_name"])
    else:
        s, a, c = cluster_for_name(bde_name, ac, dealers, summary)
        clust_df = c
        centers = c.copy()
    metrics = {}
    metrics["dealers"] = int(pick["id_dealer_outlet"].nunique()) if not pick.empty else 0
    metrics["active_dealers"] = int(pick[pick["active_dse"].fillna(0) > 0]["id_dealer_outlet"].nunique()) if not pick.empty else 0
    metrics["active_dse"] = int(pick["active_dse"].fillna(0).sum()) if not pick.empty else 0
    metrics["avg_weekly"] = float(pick["avg_weekly_visits"].fillna(0).mean()) if not pick.empty else 0.0
    return summary, centers, avail_df_merge, pick, metrics
