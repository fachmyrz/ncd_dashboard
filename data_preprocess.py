import pandas as pd
import numpy as np
import geopy.distance
from data_load import cluster_left, location_detail, df_visit as df_visit_raw, df_dealer as df_dealer_raw, running_order
from sklearn.cluster import KMeans
from kneed import KneeLocator

def parse_latlon_series(s):
    s = s.astype(str).str.replace("`","",regex=False).str.strip()
    lat = pd.to_numeric(s.str.extract(r'^\s*([\-0-9\.]+)')[0], errors="coerce")
    lon = pd.to_numeric(s.str.extract(r'[, ]\s*([\-0-9\.]+)\s*$')[0], errors="coerce")
    return lat, lon

df_dealer = df_dealer_raw.copy()
df_dealer = df_dealer[["id_dealer_outlet","brand","business_type","city","name","state","latitude","longitude"]]
df_dealer["business_type"] = df_dealer["business_type"].astype(str).str.strip()
df_dealer = df_dealer[df_dealer["business_type"].str.lower()=="car"]
df_dealer["latitude"] = df_dealer["latitude"].astype(str).str.replace("`","",regex=False).str.replace(",","",regex=False).str.strip().str.strip(".")
df_dealer["longitude"] = df_dealer["longitude"].astype(str).str.replace("`","",regex=False).str.replace(",","",regex=False).str.strip().str.strip(".")
df_dealer["latitude"] = pd.to_numeric(df_dealer["latitude"], errors="coerce")
df_dealer["longitude"] = pd.to_numeric(df_dealer["longitude"], errors="coerce")
df_dealer = df_dealer.dropna(subset=["latitude","longitude"]).reset_index(drop=True)

v = df_visit_raw.copy()
rename_map = {}
for src, dst in [
    ("Employee Name","employee_name"),
    ("Nama Karyawan","employee_name"),
    ("Client Name","client_name"),
    ("Nama Klien","client_name"),
    ("Date Time Start","date_time_start"),
    ("Tanggal Datang","date_time_start"),
    ("Date Time End","date_time_end"),
    ("Note Start","note_start"),
    ("Note End","note_end"),
    ("Longitude Start","long"),
    ("Latitude Start","lat"),
    ("Latitude & Longitude Datang","latlong"),
]:
    if src in v.columns:
        rename_map[src] = dst
v = v.rename(columns=rename_map)
if "lat" not in v.columns or "long" not in v.columns:
    if "latlong" in v.columns:
        lat_s, lon_s = parse_latlon_series(v["latlong"])
        v["lat"] = lat_s
        v["long"] = lon_s
    else:
        v["lat"] = pd.NA
        v["long"] = pd.NA
v["lat"] = pd.to_numeric(v["lat"], errors="coerce")
v["long"] = pd.to_numeric(v["long"], errors="coerce")
if "date_time_start" in v.columns:
    s = v["date_time_start"].astype(str)
    if s.str.contains("@").any():
        dates = pd.to_datetime(s.str.split("@").str[0].str.strip(), format="%d %b %Y", errors="coerce")
        times = pd.to_datetime(s.str.split("@").str[1], errors="coerce")
        v["date"] = dates.dt.date
        v["time_start"] = times.dt.time
    else:
        dt = pd.to_datetime(s, errors="coerce")
        v["date"] = dt.dt.date
        v["time_start"] = dt.dt.time
else:
    v["date"] = pd.NaT
    v["time_start"] = pd.NaT
if "date_time_end" in v.columns:
    te = pd.to_datetime(v["date_time_end"].astype(str).str.split("@").str[-1], errors="coerce")
    v["time_end"] = te.dt.time
else:
    v["time_end"] = pd.NaT
ts_end = pd.to_datetime(v["time_end"].astype(str), errors="coerce")
ts_start = pd.to_datetime(v["time_start"].astype(str), errors="coerce")
v["duration"] = (ts_end - ts_start).dt.total_seconds() / 60
df_visit = v[["employee_name","client_name","date","time_start","time_end","duration","lat","long"] + [c for c in ["Nomor Induk Karyawan","Divisi"] if c in v.columns]].copy()

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
