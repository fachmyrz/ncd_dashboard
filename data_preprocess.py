import pandas as pd
import numpy as np
import geopy.distance
from sklearn.cluster import KMeans
from kneed import KneeLocator
from datetime import datetime, timedelta
from data_load import cluster_left, location_detail, df_visit, df_dealer, sales_data, running_order

pd.options.mode.copy_on_write = True

def _to_float(x):
    try:
        s = str(x).strip().replace("`","").replace("’","").replace("“","").replace("”","")
        s = s.replace(" ", "").replace(",", "") if s.count(",")>1 else s.replace(" ", "")
        return float(s)
    except Exception:
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return np.nan

def _valid_latlon(a, b):
    try:
        if pd.isna(a) or pd.isna(b):
            return False
        la, lo = float(a), float(b)
        return np.isfinite(la) and np.isfinite(lo) and -90 <= la <= 90 and -180 <= lo <= 180
    except Exception:
        return False

def _km(a1, o1, a2, o2):
    if not _valid_latlon(a1,o1) or not _valid_latlon(a2,o2):
        return np.inf
    try:
        return geopy.distance.geodesic((float(a1),float(o1)),(float(a2),float(o2))).km
    except Exception:
        return np.inf

def _pick(df, names, default=None):
    for n in names:
        if n in df.columns:
            return n
    return default

def _normalize_dealers(df):
    if df.empty:
        return pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","state","latitude","longitude"])
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    idcol = _pick(d, ["id_dealer_outlet","dealer_id","id","dealerId"])
    brand = _pick(d, ["brand","Brand"])
    city = _pick(d, ["city","City"])
    name = _pick(d, ["name","dealer_name","Dealer Name"])
    lat = _pick(d, ["latitude","Latitude"])
    lon = _pick(d, ["longitude","Longitude"])
    state = _pick(d, ["state","State"])
    d = d[[c for c in [idcol,brand,city,name,state,lat,lon] if c]].rename(columns={idcol:"id_dealer_outlet",brand:"brand",city:"city",name:"name",state:"state",lat:"latitude",lon:"longitude"})
    d["business_type"] = "Car"
    d = d.dropna(subset=["id_dealer_outlet","brand","city","name","latitude","longitude"]).copy()
    d["latitude"] = d["latitude"].apply(_to_float)
    d["longitude"] = d["longitude"].astype(str).str.strip(".").apply(_to_float)
    d = d[d.apply(lambda r: _valid_latlon(r["latitude"], r["longitude"]), axis=1)]
    d["id_dealer_outlet"] = pd.to_numeric(d["id_dealer_outlet"], errors="coerce").astype("Int64")
    return d.reset_index(drop=True)

def _normalize_visits(df):
    if df.empty:
        return pd.DataFrame(columns=["employee_name","client_name","date","time_start","time_end","lat","long","duration"])
    v = df.copy()
    v.columns = [c.strip() for c in v.columns]
    emp = _pick(v, ["Employee Name","employee_name","nama karyawan","Nama Karyawan"])
    cli = _pick(v, ["Client Name","client_name","Nama Klien"])
    dt_start = _pick(v, ["Date Time Start","date_time_start","Datetime Start","start_time","Start Time"])
    dt_end = _pick(v, ["Date Time End","date_time_end","Datetime End","end_time","End Time"])
    lon_s = _pick(v, ["Longitude Start","long","lon","lng","Longitude"])
    lat_s = _pick(v, ["Latitude Start","lat","latitude","Latitude"])
    v = v[[c for c in [emp,cli,dt_start,dt_end,lon_s,lat_s] if c]].rename(columns={emp:"employee_name",cli:"client_name",dt_start:"date_time_start",dt_end:"date_time_end",lon_s:"long",lat_s:"lat"})
    v["time_start"] = v["date_time_start"].astype(str).apply(lambda x: x.split("@")[1] if "@" in x else x)
    v["time_end"] = v["date_time_end"].astype(str).apply(lambda x: x.split("@")[1] if "@" in x else x)
    v["date"] = v["date_time_start"].astype(str).apply(lambda x: x.split("@")[0] if "@" in x else x)
    v["date"] = pd.to_datetime(v["date"], errors="coerce").dt.date
    v["time_start"] = pd.to_datetime(v["time_start"].astype(str), errors="coerce").dt.time
    v["time_end"] = pd.to_datetime(v["time_end"].astype(str), errors="coerce").dt.time
    td = pd.to_datetime(v["time_end"].astype(str), errors="coerce") - pd.to_datetime(v["time_start"].astype(str), errors="coerce")
    v["duration"] = td.dt.total_seconds()/60
    v["lat"] = v["lat"].apply(_to_float)
    v["long"] = v["long"].apply(_to_float)
    return v

def _normalize_orders(df):
    if df.empty:
        return pd.DataFrame(columns=["id_dealer_outlet","revenue","order_date"])
    o = df.copy()
    o.columns = [c.strip() for c in o.columns]
    did = _pick(o, ["dealer_id","id_dealer_outlet","Dealer Id","dealerId"])
    amt = _pick(o, ["total_paid_after_tax","Amount After Tax","amount_after_tax","amount"])
    datec = _pick(o, ["order_date","created_at","date","Order Date"])
    o = o[[c for c in [did,amt,datec] if c]].rename(columns={did:"id_dealer_outlet",amt:"revenue",datec:"order_date"})
    o["id_dealer_outlet"] = pd.to_numeric(o["id_dealer_outlet"], errors="coerce").astype("Int64")
    o["revenue"] = pd.to_numeric(o["revenue"], errors="coerce")
    o["order_date"] = pd.to_datetime(o["order_date"], errors="coerce")
    return o

df_dealer = _normalize_dealers(df_dealer)
df_visit = _normalize_visits(df_visit)
sales_data = _normalize_orders(sales_data)

def get_summary_data(pick_date="2024-11-01"):
    if df_visit.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = df_visit[df_visit["date"] >= pd.to_datetime(pick_date).date()].copy()
    summary["lat"] = summary["lat"].astype(float)
    summary["long"] = summary["long"].astype(float)
    data = []
    for dates in summary["date"].dropna().unique():
        for name in summary["employee_name"].dropna().unique():
            temp = summary[(summary.employee_name == name) & (summary["date"] == dates)][["date","employee_name","lat","long","time_start","time_end"]].reset_index(drop=True)
            if len(temp) > 1:
                dist, time_between = [], []
                for i in range(len(temp)-1):
                    dist.append(round(_km(temp.loc[i+1,"lat"],temp.loc[i+1,"long"], temp.loc[i,"lat"],temp.loc[i,"long"]),2))
                    time_between.append((pd.to_datetime(str(temp.loc[i+1,"time_start"])) - pd.to_datetime(str(temp.loc[i,"time_start"]))).total_seconds()/60)
                avg_speed = round(sum(dist)/sum(time_between),2) if sum(time_between) != 0 else 0
                data.append([dates,name,len(temp),round(np.mean(dist),2),round(np.mean(time_between),2),avg_speed])
            else:
                data.append([dates,name,len(temp),0.0,0.0,0.0])
    cols = ["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"]
    data = pd.DataFrame(data, columns=cols)
    data["month_year"] = pd.to_datetime(data["date"]).astype("datetime64[M]").astype(str)
    return summary, data

summary, data_sum = get_summary_data()

filter_data = []
for name in summary.employee_name.dropna().unique():
    lat_long = summary[summary.employee_name == name][["lat","long"]]
    if lat_long.empty:
        continue
    min_lat = lat_long["lat"].min()
    max_lat = lat_long["lat"].max()
    min_long = lat_long["long"].min()
    max_long = lat_long["long"].max()
    lat_ = geopy.distance.geodesic((max_lat,min_long),(min_lat,min_long)).km if np.isfinite(min_lat) and np.isfinite(max_lat) else 0
    long_ = geopy.distance.geodesic((min_lat,max_long),(min_lat,min_long)).km if np.isfinite(min_long) and np.isfinite(max_long) else 0
    area = lat_ * long_
    filter_data.append([name,min_lat,max_lat,min_long,max_long,area])

area_coverage = pd.DataFrame(data=filter_data,columns=["employee_name","min_lat","max_lat","min_long","max_long","area"])
for c in ["min_lat","min_long","max_lat","max_long"]:
    area_coverage[c] = area_coverage[c].astype(float)

def _cluster_for_sales(name):
    box = area_coverage[area_coverage.employee_name == name]
    if box.empty:
        return None, None, None
    get_dealer = df_dealer[(df_dealer.latitude.between(box.min_lat.values[0], box.max_lat.values[0])) & (df_dealer.longitude.between(box.min_long.values[0], box.max_long.values[0]))]
    sum_ = summary[summary.employee_name == name][["date","client_name","lat","long"]].rename(columns={"lat":"latitude","long":"longitude"}).copy()
    sum_["sales_name"] = name
    avail_ = get_dealer[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
    avail_["tag"] = "avail"
    avail_["sales_name"] = name
    if len(sum_) >= 2:
        X = list(zip(sum_["latitude"], sum_["longitude"]))
        wcss = []
        for i in range(4, min(9, len(sum_))):
            wcss.append(KMeans(n_clusters=i, n_init="auto").fit(X).inertia_)
        knee = KneeLocator(range(4, min(9, len(sum_))), wcss, curve="convex", direction="decreasing")
        n_cluster = knee.elbow if knee.elbow is not None else 4
        km = KMeans(n_clusters=n_cluster, n_init="auto").fit(X)
        sum_["cluster"] = km.labels_
        centers = pd.DataFrame(km.cluster_centers_, columns=["latitude","longitude"])
        for i in range(len(km.cluster_centers_)):
            latc, lonc = km.cluster_centers_[i]
            avail_[f"dist_center_{i}"] = avail_.apply(lambda r: _km(latc, lonc, r["latitude"], r["longitude"]), axis=1)
    else:
        sum_["cluster"] = 0
        latm = float(sum_["latitude"].mean()) if len(sum_) else np.nan
        lonm = float(sum_["longitude"].mean()) if len(sum_) else np.nan
        centers = pd.DataFrame([[latm, lonm]], columns=["latitude","longitude"])
        avail_["dist_center_0"] = avail_.apply(lambda r: _km(latm, lonm, r["latitude"], r["longitude"]), axis=1)
    centers["sales_name"] = name
    centers["cluster"] = range(len(centers))
    return sum_, avail_, centers

sum_data, avail_data, cluster_center = [], [], []
for nm in area_coverage.employee_name.dropna().unique():
    s, a, c = _cluster_for_sales(nm)
    if s is None:
        continue
    sum_data.append(s)
    avail_data.append(a)
    cluster_center.append(c)

sum_df = pd.concat(sum_data) if sum_data else pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
avail_df = pd.concat(avail_data) if avail_data else pd.DataFrame()
clust_df = pd.concat(cluster_center) if cluster_center else pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])

active_order = running_order[["Dealer Id","Dealer Name","IsActive","End Date"]].rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name"})
active_order = active_order[active_order["IsActive"] == "1"].copy()
active_order["End Date"] = pd.to_datetime(active_order["End Date"], errors="coerce")
active_order["id_dealer_outlet"] = pd.to_numeric(active_order["id_dealer_outlet"], errors="coerce").astype("Int64")
active_order_group = active_order.groupby(["id_dealer_outlet","dealer_name"], as_index=False)["End Date"].min().rename(columns={"End Date":"nearest_end_date"})

run_order = running_order[["Dealer Id","Dealer Name","LMS Id","IsActive"]].rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
run_order["id_dealer_outlet"] = pd.to_numeric(run_order["id_dealer_outlet"], errors="coerce").astype("Int64")
run_order["active_dse"] = pd.to_numeric(run_order["active_dse"], errors="coerce").astype("Int64")
run_grouped = run_order.groupby(["id_dealer_outlet","dealer_name"], as_index=False).agg(joined_dse=("joined_dse","count"), active_dse=("active_dse","sum"))
run_order_group = pd.merge(run_grouped, active_order_group, how="left", on=["id_dealer_outlet","dealer_name"])
if not avail_df.empty:
    avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce").astype("Int64")

if not avail_df.empty and avail_df.shape[1] > 9:
    min_values = avail_df.fillna(1_000_000_000).iloc[:, 9:].min(axis=1)
    avail_df.iloc[:, 9:] = avail_df.iloc[:, 9:].where(avail_df.iloc[:, 9:].eq(min_values, axis=0), np.nan)

avail_df_merge = pd.merge(avail_df, run_order_group.drop(columns=["dealer_name"]), how="left", on="id_dealer_outlet")
if not location_detail.empty:
    avail_df_merge = pd.merge(avail_df_merge, location_detail[["City","Cluster"]].rename(columns={"City":"city","Cluster":"cluster"}), how="left", on="city")

if not cluster_left.empty:
    nc = cluster_left[cluster_left.get("Category","Car").astype(str).str.lower().eq("car")].replace({"CHERY":"Chery","Kia":"KIA"}).rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"})
    avail_df_merge = pd.merge(avail_df_merge, nc[["cluster","brand","daily_gen","daily_need","delta","availability"]], how="left", on=["brand","cluster"])

avail_df_merge["tag"] = np.where(avail_df_merge.get("nearest_end_date").isna(), "Not Active", "Active")

def _revenue_agg(orders):
    if orders.empty:
        return pd.DataFrame(columns=["id_dealer_outlet","revenue_total","revenue_mtd","revenue_last_30d"])
    o = orders.copy()
    today = pd.Timestamp.today().normalize()
    start_month = today.replace(day=1)
    rev_total = o.groupby("id_dealer_outlet", as_index=False)["revenue"].sum().rename(columns={"revenue":"revenue_total"})
    mtd = o[o["order_date"]>=start_month].groupby("id_dealer_outlet", as_index=False)["revenue"].sum().rename(columns={"revenue":"revenue_mtd"})
    last30 = o[o["order_date"]>=today-pd.Timedelta(days=30)].groupby("id_dealer_outlet", as_index=False)["revenue"].sum().rename(columns={"revenue":"revenue_last_30d"})
    return rev_total.merge(mtd, on="id_dealer_outlet", how="left").merge(last30, on="id_dealer_outlet", how="left")

rev = _revenue_agg(sales_data)
if not rev.empty:
    avail_df_merge = avail_df_merge.merge(rev, on="id_dealer_outlet", how="left")
