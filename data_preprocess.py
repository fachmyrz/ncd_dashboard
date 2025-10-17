import pandas as pd
import numpy as np
import geopy.distance
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from kneed import KneeLocator
from data_load import cluster_left, location_detail, df_visit, df_dealer, running_order

pd.options.mode.copy_on_write = True

def _to_float_series(s):
    return pd.to_numeric(
        pd.Series(s, dtype="object")
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("`", "", regex=False)
        .str.replace("’", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

def _split_at(x, part):
    x = "" if x is None else str(x)
    if "@" not in x:
        return x if part == "date" else None
    a, b = x.split("@", 1)
    return a.strip() if part == "date" else b.strip()

def clean_dealers(df):
    d = df.copy()
    d = d.rename(columns={c: c.strip() for c in d.columns})
    keep = ["id_dealer_outlet","brand","business_type","city","name","state","latitude","longitude"]
    d = d[[c for c in keep if c in d.columns]]
    if "business_type" not in d.columns:
        d["business_type"] = "Car"
    d = d[d["business_type"].astype(str).str.strip().str.lower().eq("car")]
    d["latitude"] = _to_float_series(d["latitude"])
    d["longitude"] = _to_float_series(d["longitude"])
    d = d.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    d["id_dealer_outlet"] = pd.to_numeric(d["id_dealer_outlet"], errors="coerce").astype("Int64")
    d["brand"] = d["brand"].astype(str).str.strip()
    d["city"] = d["city"].astype(str).str.strip()
    d["name"] = d["name"].astype(str).str.strip()
    return d

def clean_visits(df):
    v = df.copy()
    cols_map = {}
    for c in v.columns:
        lc = str(c).strip().lower()
        if lc in ["employee name","nama karyawan"]:
            cols_map[c] = "employee_name"
        if lc in ["client name","nama klien","client"]:
            cols_map[c] = "client_name"
        if lc in ["date time start","tanggal datang","waktu datang"]:
            cols_map[c] = "date_time_start"
        if lc in ["date time end","waktu selesai"]:
            cols_map[c] = "date_time_end"
        if lc in ["longitude start","longitude datang","long datang"]:
            cols_map[c] = "long"
        if lc in ["latitude start","latitude datang","lat datang"]:
            cols_map[c] = "lat"
        if lc in ["nomor induk karyawan","nik"]:
            cols_map[c] = "nomor_induk_karyawan"
        if lc in ["divisi","department"]:
            cols_map[c] = "divisi"
    v = v.rename(columns=cols_map)
    for req in ["employee_name","client_name","date_time_start","date_time_end","lat","long","nomor_induk_karyawan","divisi"]:
        if req not in v.columns:
            v[req] = np.nan
    v["date"] = pd.to_datetime(v["date_time_start"].apply(lambda x: _split_at(x, "date")), errors="coerce").dt.date
    v["time_start"] = pd.to_datetime(v["date_time_start"].apply(lambda x: _split_at(x, "time")), errors="coerce").dt.time
    v["time_end"] = pd.to_datetime(v["date_time_end"].apply(lambda x: _split_at(x, "time")), errors="coerce").dt.time
    v["lat"] = _to_float_series(v["lat"])
    v["long"] = _to_float_series(v["long"])
    v["duration"] = (pd.to_datetime(v["time_end"].astype(str), errors="coerce") - pd.to_datetime(v["time_start"].astype(str), errors="coerce")).dt.total_seconds() / 60
    v["employee_name"] = v["employee_name"].astype(str).str.strip()
    v["client_name"] = v["client_name"].astype(str).str.strip()
    v["nomor_induk_karyawan"] = v["nomor_induk_karyawan"].astype(str)
    v["divisi"] = v["divisi"].astype(str)
    return v

def get_summary_data(vis, pick_date="2024-11-01"):
    s = vis[vis["date"] >= pd.to_datetime(pick_date).date()].copy()
    s["lat"] = pd.to_numeric(s["lat"], errors="coerce")
    s["long"] = pd.to_numeric(s["long"], errors="coerce")
    s = s.dropna(subset=["lat","long"]).reset_index(drop=True)
    rows = []
    if not s.empty:
        for d in s["date"].unique():
            for n in s["employee_name"].dropna().unique():
                tmp = s[(s.employee_name == n) & (s["date"] == d)].reset_index(drop=True)
                tmp = tmp[["date","employee_name","lat","long","time_start","time_end"]]
                if len(tmp) > 1:
                    dist = []
                    tb = []
                    for i in range(len(tmp) - 1):
                        a = (tmp.loc[i + 1, "lat"], tmp.loc[i + 1, "long"])
                        b = (tmp.loc[i, "lat"], tmp.loc[i, "long"])
                        dist.append(round(geopy.distance.geodesic(a, b).km, 2))
                        tb.append((pd.to_datetime(str(tmp.loc[i + 1, "time_start"])) - pd.to_datetime(str(tmp.loc[i, "time_start"]))).total_seconds() / 60)
                    sp = round((sum(dist) / sum(tb)), 2) if sum(tb) != 0 else 0
                    rows.append([d, n, len(tmp), round(np.mean(dist), 2), round(np.mean(tb), 2), sp])
                else:
                    rows.append([d, n, 1, 0.0, 0.0, 0.0])
    df = pd.DataFrame(rows, columns=["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"])
    if not df.empty:
        df["month_year"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
    else:
        df["month_year"] = pd.Series(dtype="object")
    return s, df

def kmeans_for_person(df_xy):
    if len(df_xy) < 2:
        centers = np.array([[df_xy["latitude"].mean(), df_xy["longitude"].mean()]])
        labs = np.zeros(len(df_xy), dtype=int)
        return labs, centers
    X = list(zip(df_xy["latitude"], df_xy["longitude"]))
    kmax = min(8, max(4, len(df_xy)))
    wcss = []
    for k in range(4, kmax + 1):
        km = KMeans(n_clusters=k, n_init="auto").fit(X)
        wcss.append(km.inertia_)
    knee = KneeLocator(range(4, kmax + 1), wcss, curve="convex", direction="decreasing")
    n_cluster = knee.elbow if getattr(knee, "elbow", None) else min(4, kmax)
    km = KMeans(n_clusters=n_cluster, n_init="auto")
    km.fit(X)
    return km.labels_, km.cluster_centers_

def build_core_frames(vis, dealers):
    s_rows = []
    a_rows = []
    c_rows = []
    if vis.empty:
        return pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"]), pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","latitude","longitude","tag","sales_name"]), pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])
    for n in vis.employee_name.dropna().unique():
        s = vis[vis.employee_name == n][["date","client_name","lat","long"]].rename(columns={"lat":"latitude","long":"longitude"}).copy()
        s["sales_name"] = n
        if s[["latitude","longitude"]].dropna().empty:
            continue
        a = dealers.copy()
        a["tag"] = "avail"
        a["sales_name"] = n
        if len(s) >= 2:
            labs, centers = kmeans_for_person(s)
            s["cluster"] = labs
            for i in range(len(centers)):
                c_lat, c_lon = centers[i, 0], centers[i, 1]
                a[f"dist_center_{i}"] = a.apply(lambda r: geopy.distance.geodesic((c_lat, c_lon), (r.latitude, r.longitude)).km, axis=1)
        else:
            s["cluster"] = 0
            centers = np.array([[s["latitude"].mean(), s["longitude"].mean()]])
            a["dist_center_0"] = a.apply(lambda r: geopy.distance.geodesic((centers[0, 0], centers[0, 1]), (r.latitude, r.longitude)).km, axis=1)
        cl = pd.DataFrame(centers, columns=["latitude","longitude"])
        cl["sales_name"] = n
        cl["cluster"] = range(len(centers))
        s_rows.append(s)
        a_rows.append(a)
        c_rows.append(cl)
    sum_df = pd.concat(s_rows).reset_index(drop=True) if s_rows else pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
    avail_df = pd.concat(a_rows).reset_index(drop=True) if a_rows else pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","latitude","longitude","tag","sales_name"])
    clust_df = pd.concat(c_rows).reset_index(drop=True) if c_rows else pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])
    return sum_df, avail_df, clust_df

def build_orders(ro):
    active_order = ro.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","IsActive":"is_active","End Date":"end_date"})
    active_order["end_date"] = pd.to_datetime(active_order["end_date"], errors="coerce")
    active_order["id_dealer_outlet"] = pd.to_numeric(active_order["id_dealer_outlet"], errors="coerce").astype("Int64")
    on = active_order[active_order["is_active"].astype(str) == "1"].copy()
    ao_group = on.groupby(["id_dealer_outlet","dealer_name"], dropna=False).agg({"end_date":"min"}).reset_index().rename(columns={"end_date":"nearest_end_date"})
    run_order = ro.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
    run_order["id_dealer_outlet"] = pd.to_numeric(run_order["id_dealer_outlet"], errors="coerce").astype("Int64")
    run_order["active_dse"] = pd.to_numeric(run_order["active_dse"], errors="coerce").astype("Int64")
    run_order = run_order.dropna(subset=["id_dealer_outlet"])
    gr = run_order.groupby(["id_dealer_outlet","dealer_name"], dropna=False).agg({"joined_dse":"count","active_dse":"sum"}).reset_index()
    gr = gr.merge(ao_group, how="left", on=["id_dealer_outlet","dealer_name"])
    return gr

def build_availability(avail_df, grouped_ro, loc_det, need_clust):
    df = avail_df.copy()
    if "id_dealer_outlet" in df.columns:
        df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce").astype("Int64")
    dist_cols = [c for c in df.columns if c.startswith("dist_center_")]
    if dist_cols:
        vals = df[dist_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        mins = vals.min(axis=1)
        for c in dist_cols:
            vc = pd.to_numeric(df[c], errors="coerce").astype(float)
            mask = vc.sub(mins).abs() <= 1e-9
            df[c] = np.where(mask, vc, np.nan)
    out = df.merge(grouped_ro, how="left", on="id_dealer_outlet")
    ld = loc_det.rename(columns={"City":"city","Cluster":"cluster"})
    out = out.merge(ld[["city","cluster"]], how="left", on="city")
    nc = need_clust.replace({"CHERY":"Chery","Kia":"KIA"}).rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability","Category":"category"})
    nc = nc[nc["category"].astype(str).str.strip().eq("Car")]
    out = out.merge(nc[["cluster","brand","daily_gen","daily_need","delta","availability"]], how="left", on=["brand","cluster"])
    out["tag"] = np.where(out["nearest_end_date"].isna(), "Not Active", "Active")
    return out

df_dealer = clean_dealers(df_dealer)
df_visit = clean_visits(df_visit)
df_visit = df_visit[~df_visit["nomor_induk_karyawan"].astype(str).str.contains("deleted-", case=False, na=False)]
df_visit = df_visit[~df_visit["divisi"].astype(str).str.contains("trainer", case=False, na=False)]
summary, data_sum = get_summary_data(df_visit)
sum_df, avail_df, clust_df = build_core_frames(df_visit, df_dealer)
grouped_run_order = build_orders(running_order)
avail_df_merge = build_availability(avail_df, grouped_run_order, location_detail, cluster_left)
