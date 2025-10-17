import re
import pandas as pd
import numpy as np
import geopy.distance
from sklearn.cluster import KMeans
from kneed import KneeLocator
from data_load import get_sources

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

def _split_latlon(col):
    if pd.isna(col):
        return np.nan, np.nan
    s = str(col).strip().replace(";", ",")
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', s)
    if not m:
        return np.nan, np.nan
    return _to_float(m.group(1)), _to_float(m.group(2))

def _normalize_dealers(df):
    if df.empty:
        return pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","state","latitude","longitude"])
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    idcol = _pick(d, ["id_dealer_outlet","dealer_id","id","dealerId"])
    brand = _pick(d, ["brand","Brand"])
    city = _pick(d, ["city","City"])
    name = _pick(d, ["name","dealer_name","Dealer Name"])
    lat = _pick(d, ["latitude","Latitude","lat","Lat"])
    lon = _pick(d, ["longitude","Longitude","long","lon","Lng"])
    state = _pick(d, ["state","State","province"])
    use_cols = [c for c in [idcol,brand,city,name,state,lat,lon] if c]
    d = d[use_cols].rename(columns={idcol:"id_dealer_outlet",brand:"brand",city:"city",name:"name",state:"state",lat:"latitude",lon:"longitude"})
    d["business_type"] = "Car"
    d["latitude"] = d["latitude"].apply(_to_float)
    d["longitude"] = d["longitude"].astype(str).str.strip(".").apply(_to_float)
    d = d.dropna(subset=["id_dealer_outlet","brand","city","name","latitude","longitude"])
    d = d[d.apply(lambda r: _valid_latlon(r["latitude"], r["longitude"]), axis=1)]
    d["id_dealer_outlet"] = pd.to_numeric(d["id_dealer_outlet"], errors="coerce").astype("Int64")
    return d.reset_index(drop=True)

def _normalize_visits(df):
    if df.empty:
        return pd.DataFrame(columns=["employee_name","client_name","date","time_start","time_end","lat","long","duration"])
    v = df.copy()
    v.columns = [c.strip() for c in v.columns]
    emp = _pick(v, ["Employee Name","employee_name","Nama Karyawan","nama karyawan"])
    cli = _pick(v, ["Client Name","client_name","Nama Klien"])
    dt_start = _pick(v, ["Date Time Start","date_time_start","Datetime Start","start_time","Start Time","Tanggal Datang","Date Start"])
    dt_end = _pick(v, ["Date Time End","date_time_end","Datetime End","end_time","End Time","Tanggal Pulang","Date End"])
    lon_s = _pick(v, ["Longitude Start","Longitude"])
    lat_s = _pick(v, ["Latitude Start","Latitude"])
    latlon = _pick(v, ["Latitude & Longitude Datang","Latitude & Longitude","Latitude & Longitude Start"])
    use_cols = [c for c in [emp,cli,dt_start,dt_end,lon_s,lat_s,latlon] if c]
    v = v[use_cols].rename(columns={
        emp:"employee_name", cli:"client_name",
        dt_start:"date_time_start", dt_end:"date_time_end",
        lon_s:"long", lat_s:"lat", latlon:"latlon"
    })
    if "latlon" in v.columns:
        lat_series, lon_series = zip(*v["latlon"].apply(_split_latlon))
        v["lat"] = pd.Series(lat_series, index=v.index)
        v["long"] = pd.Series(lon_series, index=v.index)
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
    out_cols = ["employee_name","client_name","date","time_start","time_end","lat","long","duration"]
    for c in out_cols:
        if c not in v.columns:
            v[c] = pd.NA
    return v[out_cols]

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

def get_summary_data(visits, pick_date="2024-11-01"):
    if visits.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = visits[visits["date"] >= pd.to_datetime(pick_date).date()].copy()
    summary["lat"] = summary["lat"].astype(float)
    summary["long"] = summary["long"].astype(float)
    rows = []
    for dt in summary["date"].dropna().unique():
        for nm in summary["employee_name"].dropna().unique():
            temp = summary[(summary["employee_name"]==nm) & (summary["date"]==dt)][["date","employee_name","lat","long","time_start","time_end"]].reset_index(drop=True)
            if len(temp) > 1:
                dists, gaps = [], []
                for i in range(len(temp)-1):
                    dists.append(round(_km(temp.loc[i+1,"lat"],temp.loc[i+1,"long"], temp.loc[i,"lat"],temp.loc[i,"long"]),2))
                    gaps.append((pd.to_datetime(str(temp.loc[i+1,"time_start"])) - pd.to_datetime(str(temp.loc[i,"time_start"]))).total_seconds()/60)
                sp = round(sum(dists)/sum(gaps),2) if sum(gaps)!=0 else 0
                rows.append([dt,nm,len(temp),round(np.mean(dists),2),round(np.mean(gaps),2),sp])
            else:
                rows.append([dt,nm,len(temp),0.0,0.0,0.0])
    data = pd.DataFrame(rows, columns=["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"])
    data["month_year"] = pd.to_datetime(data["date"]).astype("datetime64[M]").astype(str)
    return summary, data

def compute_clusters(summary, dealers):
    if dealers.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if summary.empty:
        a = dealers[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
        a["tag"] = "avail"
        a["sales_name"] = "All"
        a["dist_center_0"] = 0.0
        c = pd.DataFrame([{"latitude":float(a["latitude"].mean()),"longitude":float(a["longitude"].mean()),"sales_name":"All","cluster":0}])
        s = pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
        return s, a, c
    area_rows = []
    for nm in summary["employee_name"].dropna().unique():
        lat_long = summary[summary["employee_name"]==nm][["lat","long"]]
        if lat_long.empty:
            continue
        min_lat, max_lat = lat_long["lat"].min(), lat_long["lat"].max()
        min_lon, max_lon = lat_long["long"].min(), lat_long["long"].max()
        lat_km = geopy.distance.geodesic((max_lat,min_lon),(min_lat,min_lon)).km if np.isfinite(min_lat) and np.isfinite(max_lat) else 0
        lon_km = geopy.distance.geodesic((min_lat,max_lon),(min_lat,min_lon)).km if np.isfinite(min_lon) and np.isfinite(max_lon) else 0
        area_rows.append([nm,min_lat,max_lat,min_lon,max_lon,lat_km*lon_km])
    cov = pd.DataFrame(area_rows, columns=["employee_name","min_lat","max_lat","min_long","max_long","area"])
    if cov.empty:
        a = dealers[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
        a["tag"] = "avail"
        a["sales_name"] = "All"
        a["dist_center_0"] = 0.0
        c = pd.DataFrame([{"latitude":float(a["latitude"].mean()),"longitude":float(a["longitude"].mean()),"sales_name":"All","cluster":0}])
        s = pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
        return s, a, c
    sums, avs, centers = [], [], []
    for nm in cov["employee_name"].unique():
        rect = cov[cov["employee_name"]==nm].iloc[0]
        pool = dealers[(dealers["latitude"].between(rect["min_lat"], rect["max_lat"])) & (dealers["longitude"].between(rect["min_long"], rect["max_long"]))]
        s = summary[summary["employee_name"]==nm][["date","client_name","lat","long"]].rename(columns={"lat":"latitude","long":"longitude"}).copy()
        s["sales_name"] = nm
        a = pool[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
        a["tag"] = "avail"
        a["sales_name"] = nm
        if len(s) >= 2:
            X = list(zip(s["latitude"], s["longitude"]))
            wcss = []
            for k in range(4, min(9, len(s))):
                wcss.append(KMeans(n_clusters=k, n_init="auto").fit(X).inertia_)
            knee = KneeLocator(range(4, min(9, len(s))), wcss, curve="convex", direction="decreasing")
            n_cluster = knee.elbow if knee.elbow is not None else 4
            km = KMeans(n_clusters=n_cluster, n_init="auto").fit(X)
            s["cluster"] = km.labels_
            for i in range(len(km.cluster_centers_)):
                latc, lonc = km.cluster_centers_[i]
                a[f"dist_center_{i}"] = a.apply(lambda r: _km(latc, lonc, r["latitude"], r["longitude"]), axis=1)
            c = pd.DataFrame(km.cluster_centers_, columns=["latitude","longitude"])
            c["sales_name"] = nm
            c["cluster"] = range(len(km.cluster_centers_))
        else:
            s["cluster"] = 0
            latm = float(s["latitude"].mean()) if len(s) else float(pool["latitude"].mean()) if not pool.empty else np.nan
            lonm = float(s["longitude"].mean()) if len(s) else float(pool["longitude"].mean()) if not pool.empty else np.nan
            a["dist_center_0"] = a.apply(lambda r: _km(latm, lonm, r["latitude"], r["longitude"]), axis=1)
            c = pd.DataFrame([[latm, lonm, nm, 0]], columns=["latitude","longitude","sales_name","cluster"])
        sums.append(s)
        avs.append(a)
        centers.append(c)
    return pd.concat(sums) if sums else pd.DataFrame(), pd.concat(avs) if avs else pd.DataFrame(), pd.concat(centers) if centers else pd.DataFrame()

def compute_running_orders(running_order):
    if running_order.empty:
        return pd.DataFrame(columns=["id_dealer_outlet","joined_dse","active_dse","nearest_end_date","dealer_name"])
    act = running_order[["Dealer Id","Dealer Name","IsActive","End Date"]].rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name"})
    act = act[act["IsActive"]=="1"].copy()
    act["End Date"] = pd.to_datetime(act["End Date"], errors="coerce")
    act["id_dealer_outlet"] = pd.to_numeric(act["id_dealer_outlet"], errors="coerce").astype("Int64")
    ao = act.groupby(["id_dealer_outlet","dealer_name"], as_index=False)["End Date"].min().rename(columns={"End Date":"nearest_end_date"})
    ro = running_order[["Dealer Id","Dealer Name","LMS Id","IsActive"]].rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
    ro["id_dealer_outlet"] = pd.to_numeric(ro["id_dealer_outlet"], errors="coerce").astype("Int64")
    ro["active_dse"] = pd.to_numeric(ro["active_dse"], errors="coerce").astype("Int64")
    grp = ro.groupby(["id_dealer_outlet","dealer_name"], as_index=False).agg(joined_dse=("joined_dse","count"), active_dse=("active_dse","sum"))
    out = grp.merge(ao, on=["id_dealer_outlet","dealer_name"], how="left")
    return out

def compute_revenue(orders):
    if orders.empty:
        return pd.DataFrame(columns=["id_dealer_outlet","revenue_total","revenue_mtd","revenue_last_30d"])
    o = orders.copy()
    today = pd.Timestamp.today().normalize()
    start_month = today.replace(day=1)
    rev_total = o.groupby("id_dealer_outlet", as_index=False)["revenue"].sum().rename(columns={"revenue":"revenue_total"})
    mtd = o[o["order_date"]>=start_month].groupby("id_dealer_outlet", as_index=False)["revenue"].sum().rename(columns={"revenue":"revenue_mtd"})
    last30 = o[o["order_date"]>=today-pd.Timedelta(days=30)].groupby("id_dealer_outlet", as_index=False)["revenue"].sum().rename(columns={"revenue":"revenue_last_30d"})
    return rev_total.merge(mtd, on="id_dealer_outlet", how="left").merge(last30, on="id_dealer_outlet", how="left")

def assemble_availability(avail_df, location_detail, need_cluster, run_group, revenue, dealers):
    base = avail_df.copy()
    if base.empty and not dealers.empty:
        base = dealers[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
        base["tag"] = "avail"
        base["sales_name"] = "All"
        base["dist_center_0"] = 0.0
    if base.empty:
        return pd.DataFrame()
    base["id_dealer_outlet"] = pd.to_numeric(base["id_dealer_outlet"], errors="coerce").astype("Int64")
    if base.shape[1] > 9:
        mins = base.fillna(1e12).iloc[:, 9:].min(axis=1)
        base.iloc[:, 9:] = base.iloc[:, 9:].where(base.iloc[:, 9:].eq(mins, axis=0), np.nan)
    df = base.merge(run_group.drop(columns=["dealer_name"], errors="ignore"), on="id_dealer_outlet", how="left")
    if not location_detail.empty:
        loc = location_detail.rename(columns={"City":"city","Cluster":"cluster"})
        df = df.merge(loc[["city","cluster"]], on="city", how="left")
    if not need_cluster.empty:
        nc = need_cluster[need_cluster.get("Category","Car").astype(str).str.lower().eq("car")].rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"})
        nc["brand"] = nc["brand"].replace({"CHERY":"Chery","Kia":"KIA"})
        df = df.merge(nc[["cluster","brand","daily_gen","daily_need","delta","availability"]], on=["brand","cluster"], how="left")
    if revenue is not None and not revenue.empty:
        df = df.merge(revenue, on="id_dealer_outlet", how="left")
    df["tag"] = np.where(df.get("nearest_end_date").isna(),"Not Active","Active")
    return df

def compute_all():
    s = get_sources()
    dealers = _normalize_dealers(s["df_dealer"])
    visits = _normalize_visits(s["df_visit"])
    location_detail = s["location_detail"]
    running_order = s["running_order"]
    need_cluster = s["need_cluster"]
    orders = _normalize_orders(s["orders"])
    summary, _ = get_summary_data(visits)
    sum_df, avail_df, clust_df = compute_clusters(summary, dealers)
    ro_group = compute_running_orders(running_order)
    revenue = compute_revenue(orders)
    avail_df_merge = assemble_availability(avail_df, location_detail, need_cluster, ro_group, revenue, dealers)
    return {"dealers": dealers, "visits": visits, "sum_df": sum_df, "avail_df_merge": avail_df_merge, "clust_df": clust_df}
