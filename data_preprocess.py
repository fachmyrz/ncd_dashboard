import pandas as pd
import numpy as np
import geopy.distance
from data_load import cluster_left, location_detail, df_visit as df_visit_raw, df_dealer as df_dealer_raw, running_order
from sklearn.cluster import KMeans
from kneed import KneeLocator

def pick_col(df, names, default=None):
    for n in names:
        if n in df.columns:
            return n
    return default

def parse_latlon_series(s):
    s = s.astype(str).str.replace("`","",regex=False).str.strip()
    lat = pd.to_numeric(s.str.extract(r'^\s*([\-0-9\.]+)')[0], errors="coerce")
    lon = pd.to_numeric(s.str.extract(r'[, ]\s*([\-0-9\.]+)\s*$')[0], errors="coerce")
    return lat, lon

df_dealer = df_dealer_raw.copy()
df_dealer = df_dealer[["id_dealer_outlet","brand","business_type","city","name","state","latitude","longitude"]]
df_dealer = df_dealer.dropna().reset_index(drop=True)
df_dealer["business_type"] = df_dealer["business_type"].astype(str).str.strip()
df_dealer = df_dealer[df_dealer["business_type"].str.lower()=="car"]
df_dealer["latitude"] = df_dealer["latitude"].astype(str).str.replace("`","",regex=False).str.replace(",","",regex=False).str.strip().str.strip(".")
df_dealer["longitude"] = df_dealer["longitude"].astype(str).str.replace("`","",regex=False).str.replace(",","",regex=False).str.strip().str.strip(".")
df_dealer["latitude"] = pd.to_numeric(df_dealer["latitude"], errors="coerce")
df_dealer["longitude"] = pd.to_numeric(df_dealer["longitude"], errors="coerce")
df_dealer = df_dealer.dropna(subset=["latitude","longitude"]).reset_index(drop=True)

df_visit_src = df_visit_raw.copy()
emp_col = pick_col(df_visit_src, ["Employee Name","Nama Karyawan","employee_name"], None)
cli_col = pick_col(df_visit_src, ["Client Name","Nama Klien","client_name","Dealer","dealer_name"], None)
dt_start_col = pick_col(df_visit_src, ["Date Time Start","Tanggal Datang","date_time_start","Visit Datetime"], None)
dt_end_col = pick_col(df_visit_src, ["Date Time End","date_time_end"], None)
note_start_col = pick_col(df_visit_src, ["Note Start","note_start"], None)
note_end_col = pick_col(df_visit_src, ["Note End","note_end"], None)
lon_col = pick_col(df_visit_src, ["Longitude Start","long","longitude"], None)
lat_col = pick_col(df_visit_src, ["Latitude Start","lat","latitude"], None)
latlon_combo_col = pick_col(df_visit_src, ["Latitude & Longitude Datang","latlong","LatLong"], None)
keep_cols = [c for c in [emp_col,cli_col,dt_start_col,dt_end_col,note_start_col,note_end_col,lon_col,lat_col,latlon_combo_col] if c is not None]
df_visit_src = df_visit_src[keep_cols].copy()
rename_map = {}
if emp_col: rename_map[emp_col]="employee_name"
if cli_col: rename_map[cli_col]="client_name"
if dt_start_col: rename_map[dt_start_col]="date_time_start"
if dt_end_col: rename_map[dt_end_col]="date_time_end"
if note_start_col: rename_map[note_start_col]="note_start"
if note_end_col: rename_map[note_end_col]="note_end"
if lon_col: rename_map[lon_col]="long"
if lat_col: rename_map[lat_col]="lat"
if latlon_combo_col: rename_map[latlon_combo_col]="latlong"
df_visit_src = df_visit_src.rename(columns=rename_map)
if "lat" not in df_visit_src.columns or "long" not in df_visit_src.columns:
    if "latlong" in df_visit_src.columns:
        lat_s, lon_s = parse_latlon_series(df_visit_src["latlong"])
        df_visit_src["lat"] = lat_s
        df_visit_src["long"] = lon_s
    else:
        df_visit_src["lat"] = pd.NA
        df_visit_src["long"] = pd.NA
df_visit_src["lat"] = pd.to_numeric(df_visit_src["lat"], errors="coerce")
df_visit_src["long"] = pd.to_numeric(df_visit_src["long"], errors="coerce")
if "date_time_start" in df_visit_src.columns:
    s = df_visit_src["date_time_start"].astype(str)
    if s.str.contains("@").any():
        date_part = s.apply(lambda x: x.split("@")[0] if "@" in x else x).str.strip()
        time_part = s.apply(lambda x: x.split("@")[1] if "@" in x else "")
        df_visit_src["date"] = pd.to_datetime(date_part, format="%d %b %Y", errors="coerce").dt.date
        df_visit_src["time_start"] = pd.to_datetime(time_part, errors="coerce").dt.time
    else:
        dt = pd.to_datetime(s, errors="coerce")
        df_visit_src["date"] = dt.dt.date
        df_visit_src["time_start"] = dt.dt.time
else:
    dt_alt = pick_col(df_visit_src, ["visit_datetime","Tanggal Datang"], None)
    if dt_alt:
        dt = pd.to_datetime(df_visit_src[dt_alt].astype(str), errors="coerce")
        df_visit_src["date"] = dt.dt.date
        df_visit_src["time_start"] = dt.dt.time
    else:
        df_visit_src["date"] = pd.NaT
        df_visit_src["time_start"] = pd.NaT
if "date_time_end" in df_visit_src.columns:
    t = pd.to_datetime(df_visit_src["date_time_end"].astype(str).str.split("@").str[-1], errors="coerce")
    df_visit_src["time_end"] = t.dt.time
else:
    df_visit_src["time_end"] = pd.NaT
ts_end = pd.to_datetime(df_visit_src["time_end"].astype(str), errors="coerce")
ts_start = pd.to_datetime(df_visit_src["time_start"].astype(str), errors="coerce")
df_visit_src["duration"] = (ts_end - ts_start).dt.total_seconds() / 60
df_visit = df_visit_src[["employee_name","client_name","date","time_start","time_end","duration","lat","long"]].copy()

def get_summary_data(pick_date="2024-11-01"):
    summary = df_visit[df_visit["date"] >= pd.to_datetime(pick_date).date()].copy()
    summary["lat"] = pd.to_numeric(summary["lat"], errors="coerce")
    summary["long"] = pd.to_numeric(summary["long"], errors="coerce")
    summary = summary.dropna(subset=["lat","long"])
    summary.reset_index(drop=True,inplace=True)
    rows = []
    for d in summary["date"].dropna().unique():
        for n in summary["employee_name"].dropna().unique():
            temp = summary[(summary.employee_name==n)&(summary["date"]==d)].reset_index(drop=True)
            temp = temp[["date","employee_name","lat","long","time_start","time_end"]]
            if len(temp)>1:
                dist = []
                tb = []
                for i in range(len(temp)-1):
                    dist.append(round(geopy.distance.geodesic((temp.loc[i+1,"lat"],temp.loc[i+1,"long"]),(temp.loc[i,"lat"],temp.loc[i,"long"])).km,2))
                    t1 = pd.to_datetime(str(temp.loc[i,"time_start"]), errors="coerce")
                    t2 = pd.to_datetime(str(temp.loc[i+1,"time_start"]), errors="coerce")
                    tb.append((t2-t1).total_seconds()/60 if pd.notna(t1) and pd.notna(t2) else 0)
                spd = round(sum(dist)/sum(tb),2) if sum(tb)!=0 else 0
                rows.append([d,n,len(temp),round(np.mean(dist),2),round(np.mean(tb),2),spd])
            else:
                rows.append([d,n,len(temp),0.0,0.0,0.0])
    df = pd.DataFrame(rows, columns=["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"])
    df["month_year"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    return summary, df

summary, data_sum = get_summary_data()
flt = []
for n in summary["employee_name"].dropna().unique():
    ll = summary[summary.employee_name==n][["lat","long"]].dropna()
    if ll.empty:
        continue
    mi_la = ll["lat"].min()
    ma_la = ll["lat"].max()
    mi_lo = ll["long"].min()
    ma_lo = ll["long"].max()
    lat_km = geopy.distance.geodesic((ma_la,mi_lo),(mi_la,mi_lo)).km
    lon_km = geopy.distance.geodesic((mi_la,ma_lo),(mi_la,mi_lo)).km
    area = lat_km*lon_km
    flt.append([n,mi_la,ma_la,mi_lo,ma_lo,area])
area_coverage = pd.DataFrame(flt, columns=["employee_name","min_lat","max_lat","min_long","max_long","area"])
for c in ["min_lat","min_long","max_lat","max_long"]:
    area_coverage[c] = pd.to_numeric(area_coverage[c], errors="coerce")

def get_distance_dealer(cluster,lat,long):
    return geopy.distance.geodesic((kmeans.cluster_centers_[cluster,0],kmeans.cluster_centers_[cluster,1]),(lat,long)).km

sum_data = []
avail_data = []
cluster_center = []
for n in area_coverage["employee_name"].dropna().unique():
    box = area_coverage[area_coverage.employee_name==n]
    if box.empty:
        continue
    dealers = df_dealer[
        df_dealer["latitude"].between(box.min_lat.values[0], box.max_lat.values[0]) &
        df_dealer["longitude"].between(box.min_long.values[0], box.max_long.values[0])
    ]
    s = summary[summary.employee_name==n][["date","client_name","lat","long"]].copy()
    s = s.rename(columns={"lat":"latitude","long":"longitude"})
    s["sales_name"] = n
    a = dealers[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
    a["tag"] = "avail"
    a["sales_name"] = n
    if len(s)>=2:
        X = list(zip(s["latitude"], s["longitude"]))
        cand = list(range(4, min(9, len(s)+1))) or [2,3,4]
        wcss = []
        for k in cand:
            kmeans = KMeans(n_clusters=k, n_init=10).fit(X)
            wcss.append(kmeans.inertia_)
        start = cand[0]
        knee = KneeLocator(range(start, start+len(wcss)), wcss, curve="convex", direction="decreasing")
        k_best = knee.elbow if knee.elbow is not None else (4 if len(s)>=4 else 2)
        kmeans = KMeans(n_clusters=k_best, n_init=10)
        kmeans.fit(X)
        s["cluster"] = kmeans.labels_
        for i in range(len(kmeans.cluster_centers_)):
            a[f"dist_center_{i}"] = a.apply(lambda r: get_distance_dealer(i, r.latitude, r.longitude), axis=1)
    else:
        s["cluster"] = 0
        kmeans = KMeans(n_clusters=1, n_init=10).fit([[0,0],[0,0]])
    c = pd.DataFrame(kmeans.cluster_centers_, columns=["latitude","longitude"])
    c["sales_name"] = n
    c["cluster"] = range(len(kmeans.cluster_centers_))
    cluster_center.append(c)
    avail_data.append(a)
    sum_data.append(s)

sum_df = pd.concat(sum_data) if sum_data else pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
avail_df = pd.concat(avail_data) if avail_data else pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","latitude","longitude","tag","sales_name"])
clust_df = pd.concat(cluster_center) if cluster_center else pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])

active_order = running_order[["Dealer Id","Dealer Name","IsActive","End Date"]].copy()
active_order = active_order[active_order["IsActive"]=="1"]
active_order["End Date"] = pd.to_datetime(active_order["End Date"], errors="coerce")
active_order["Dealer Id"] = pd.to_numeric(active_order["Dealer Id"], errors="coerce").astype("Int64")
ao_group = active_order.groupby(["Dealer Id","Dealer Name"]).agg({"End Date":"min"}).reset_index()
ao_group = ao_group.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","End Date":"nearest_end_date"})

run_order = running_order[["Dealer Id","Dealer Name","LMS Id","IsActive"]].copy()
run_order = run_order.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
run_order["id_dealer_outlet"] = pd.to_numeric(run_order["id_dealer_outlet"], errors="coerce").astype("Int64")
run_order["active_dse"] = pd.to_numeric(run_order["active_dse"], errors="coerce").astype("Int64")
run_order = run_order[~run_order["id_dealer_outlet"].isna()]
grouped_run_order = run_order.groupby(["id_dealer_outlet","dealer_name"]).agg({"joined_dse":"count","active_dse":"sum"}).reset_index()
grouped_run_order = grouped_run_order.merge(ao_group, how="left", on="id_dealer_outlet")
grouped_run_order["id_dealer_outlet"] = pd.to_numeric(grouped_run_order["id_dealer_outlet"], errors="coerce").astype("Int64")
avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce").astype("Int64")

dist_cols = [c for c in avail_df.columns if str(c).startswith("dist_center_")]
if dist_cols:
    mv = avail_df[dist_cols].apply(pd.to_numeric, errors="coerce").fillna(1e12).min(axis=1)
    mask = avail_df[dist_cols].apply(pd.to_numeric, errors="coerce").eq(mv, axis=0)
    avail_df[dist_cols] = avail_df[dist_cols].where(mask, np.nan)

avail_df_merge = avail_df.merge(grouped_run_order, how="left", on="id_dealer_outlet")
ld = location_detail.rename(columns={"City":"city","Cluster":"cluster"})
avail_df_merge = avail_df_merge.merge(ld[["city","cluster"]], how="left", on="city")

cl_map = cluster_left.copy()
cl_map = cl_map.replace({"CHERY":"Chery","Kia":"KIA"})
cl_map = cl_map[cl_map["Category"].astype(str).str.lower()=="car"]
cl_map = cl_map[["Cluster","Brand","Daily_Gen","Daily_Need","Delta","Tag"]].rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"})
avail_df_merge = avail_df_merge.merge(cl_map, how="left", on=["brand","cluster"])
avail_df_merge["tag"] = np.where(avail_df_merge["nearest_end_date"].isna(),"Not Active","Active")
