import pandas as pd
import numpy as np
import geopy.distance
from data_load import cluster_left, location_detail, df_visit, df_dealer, running_order
from sklearn.cluster import KMeans
from kneed import KneeLocator

jabodetabek_list = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

def _to_float(x):
    try:
        s = str(x).strip().replace("`","").replace(",","")
        return float(s)
    except:
        return np.nan

def _parse_latlon_pair(s):
    s = str(s)
    if "," in s:
        a,b = s.split(",",1)
        return _to_float(a), _to_float(b)
    return np.nan, np.nan

def clean_dealers(df):
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}
    idcol = cols.get("id_dealer_outlet") or cols.get("dealer_id") or list(df.columns)[0]
    brand = cols.get("brand") or "brand"
    btype = cols.get("business_type") or "business_type"
    city = cols.get("city") or "city"
    name = cols.get("name") or cols.get("dealer_name") or "name"
    state = cols.get("state") or "state"
    lat = cols.get("latitude") or "latitude"
    lon = cols.get("longitude") or "longitude"
    cl = [idcol, brand, btype, city, name, state, lat, lon]
    cl = [c for c in cl if c in df.columns]
    df = df[cl].rename(columns={idcol:"id_dealer_outlet", brand:"brand", btype:"business_type", city:"city", name:"name", state:"state", lat:"latitude", lon:"longitude"})
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].apply(_to_float)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].apply(_to_float)
    df = df[df["business_type"].astype(str).str.strip().str.title().isin(["Car"])]
    df = df.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    return df

def clean_visits(df):
    df = df.copy()
    up = {c.lower().strip(): c for c in df.columns}
    emp = up.get("nama karyawan") or up.get("employee name") or up.get("employee")
    cli = up.get("nama klien") or up.get("client name") or up.get("client")
    dt = up.get("tanggal datang") or up.get("date time start") or up.get("visit datetime")
    ll_col = up.get("latitude & longitude datang") or None
    lat_col = up.get("latitude start") or up.get("latitude") or None
    lon_col = up.get("longitude start") or up.get("longitude") or None
    nik_col = up.get("nomor induk karyawan") or up.get("nik")
    div_col = up.get("divisi") or up.get("division")
    need = [emp, cli, dt]
    need = [c for c in need if c]
    base = df.copy()
    for c in need:
        if c not in base.columns:
            base[c] = np.nan
    base = base.rename(columns={emp:"employee_name", cli:"client_name", dt:"visit_datetime"})
    if ll_col and ll_col in base.columns:
        latlon = base[ll_col].apply(_parse_latlon_pair)
        base["lat"] = latlon.apply(lambda t: t[0])
        base["lon"] = latlon.apply(lambda t: t[1])
    else:
        if lat_col in base.columns and lon_col in base.columns:
            base["lat"] = base[lat_col].apply(_to_float)
            base["lon"] = base[lon_col].apply(_to_float)
        else:
            base["lat"] = np.nan
            base["lon"] = np.nan
    if nik_col and nik_col in df.columns:
        base["nik"] = df[nik_col].astype(str)
    else:
        base["nik"] = ""
    if div_col and div_col in df.columns:
        base["division"] = df[div_col].astype(str)
    else:
        base["division"] = ""
    base["visit_datetime"] = pd.to_datetime(base["visit_datetime"], errors="coerce")
    base["date"] = base["visit_datetime"].dt.date
    base = base.dropna(subset=["date"]).reset_index(drop=True)
    base = base[~base["nik"].str.contains("deleted-", case=False, na=False)]
    base = base[~base["division"].str.contains("trainer", case=False, na=False)]
    base["employee_name"] = base["employee_name"].astype(str).str.strip()
    return base

def clean_running_order(ro):
    ro = ro.copy()
    cols = {c.lower().strip(): c for c in ro.columns}
    did = cols.get("dealer id") or cols.get("id_dealer_outlet") or "Dealer Id"
    dname = cols.get("dealer name") or "Dealer Name"
    lms = cols.get("lms id") or "LMS Id"
    isact = cols.get("isactive") or "IsActive"
    endd = cols.get("end date") or "End Date"
    for c in [did,dname,lms,isact,endd]:
        if c not in ro.columns:
            ro[c] = np.nan
    a = ro[[did,dname,isact,endd]].rename(columns={did:"id_dealer_outlet", dname:"dealer_name", isact:"IsActive", endd:"End Date"})
    a = a[a["IsActive"].astype(str).str.strip() == "1"]
    a["id_dealer_outlet"] = pd.to_numeric(a["id_dealer_outlet"], errors="coerce").astype("Int64")
    a["End Date"] = pd.to_datetime(a["End Date"], errors="coerce")
    ao_group = a.groupby(["id_dealer_outlet","dealer_name"], dropna=False).agg({"End Date":"min"}).reset_index().rename(columns={"End Date":"nearest_end_date"})
    r = ro[[did,dname,lms,isact]].rename(columns={did:"id_dealer_outlet", dname:"dealer_name", lms:"joined_dse", isact:"active_dse"})
    r["id_dealer_outlet"] = pd.to_numeric(r["id_dealer_outlet"], errors="coerce").astype("Int64")
    r["active_dse"] = pd.to_numeric(r["active_dse"], errors="coerce").astype("Int64")
    r = r.dropna(subset=["id_dealer_outlet"])
    grouped_run = r.groupby(["id_dealer_outlet","dealer_name"], dropna=False).agg({"joined_dse":"count","active_dse":"sum"}).reset_index()
    grouped_run = grouped_run.merge(ao_group, how="left", on=["id_dealer_outlet","dealer_name"])
    grouped_run["id_dealer_outlet"] = grouped_run["id_dealer_outlet"].astype(int)
    return grouped_run

def get_summary_data(visits, start_date="2024-11-01"):
    s = visits[visits["date"] >= pd.to_datetime(start_date).date()].copy()
    s["lat"] = pd.to_numeric(s["lat"], errors="coerce")
    s["lon"] = pd.to_numeric(s["lon"], errors="coerce")
    s = s.dropna(subset=["lat","lon"])
    rows = []
    for d in s["date"].unique():
        for emp in s["employee_name"].unique():
            t = s[(s["employee_name"] == emp) & (s["date"] == d)][["date","employee_name","lat","lon","visit_datetime"]].sort_values("visit_datetime").reset_index(drop=True)
            if len(t) > 1:
                dist = []
                gap = []
                for i in range(len(t)-1):
                    dist.append(round(geopy.distance.geodesic((t.loc[i+1,"lat"],t.loc[i+1,"lon"]), (t.loc[i,"lat"],t.loc[i,"lon"])).km,2))
                    gap.append((pd.to_datetime(str(t.loc[i+1,"visit_datetime"])) - pd.to_datetime(str(t.loc[i,"visit_datetime"]))).total_seconds()/60)
                spd = round(sum(dist)/sum(gap),2) if sum(gap) != 0 else 0
                rows.append([d,emp,len(t),round(np.mean(dist),2),round(np.mean(gap),2),spd])
            else:
                rows.append([d,emp,len(t),0.0,0.0,0.0])
    cols = ["date","employee_name","ctd_visit","avg_distance_km","avg_time_between_minute","avg_speed_kmpm"]
    data = pd.DataFrame(rows, columns=cols)
    data["month_year"] = data["date"].astype(str).str.slice(0,7)
    return s, data

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    v = visits.copy()
    d = dealers.copy()
    v["client_name"] = v["client_name"].astype(str).str.strip()
    d["name"] = d["name"].astype(str).str.strip()
    d["city"] = d["city"].astype(str).str.strip()
    v["city_guess"] = v["client_name"].apply(lambda s: s.split("-")[-1].strip() if "-" in s else "")
    def match_row(r):
        cn = r["client_name"]
        lat = r["lat"]
        lon = r["lon"]
        cand = d.copy()
        if pd.notna(r["city_guess"]) and r["city_guess"]:
            cand = cand[cand["city"].str.lower() == r["city_guess"].lower()]
        exact = cand[cand["name"].str.lower() == cn.lower()]
        if len(exact) == 1:
            return exact.iloc[0]["name"]
        if pd.notna(lat) and pd.notna(lon):
            if cand.empty:
                cand = d
            dist = cand.apply(lambda x: geopy.distance.geodesic((lat,lon),(x.latitude,x.longitude)).km, axis=1)
            cand = cand.assign(_km=dist)
            near = cand[cand["_km"] <= max_km].sort_values("_km")
            if not near.empty:
                return near.iloc[0]["name"]
        return np.nan
    v["matched_client"] = v.apply(match_row, axis=1)
    return v

df_dealer = clean_dealers(df_dealer)
df_visit = clean_visits(df_visit)
df_visits = assign_visits_to_dealers(df_visit, df_dealer, max_km=1.0)

summary, data_sum = get_summary_data(df_visits)

filter_data = []
for nm in summary["employee_name"].dropna().unique():
    lat_long = summary[summary["employee_name"] == nm][["lat","lon"]]
    if lat_long.empty:
        continue
    min_lat = lat_long["lat"].min()
    max_lat = lat_long["lat"].max()
    min_lon = lat_long["lon"].min()
    max_lon = lat_long["lon"].max()
    lat_km = geopy.distance.geodesic((max_lat,min_lon),(min_lat,min_lon)).km
    lon_km = geopy.distance.geodesic((min_lat,max_lon),(min_lat,min_lon)).km
    area = lat_km * lon_km
    filter_data.append([nm,min_lat,max_lat,min_lon,max_lon,area])

area_coverage = pd.DataFrame(filter_data, columns=["employee_name","min_lat","max_lat","min_lon","max_lon","area"])
for c in ["min_lat","max_lat","min_lon","max_lon"]:
    area_coverage[c] = pd.to_numeric(area_coverage[c], errors="coerce")

def get_distance_dealer(cluster, lat, lon, km_model):
    return geopy.distance.geodesic((km_model.cluster_centers_[cluster,0], km_model.cluster_centers_[cluster,1]), (lat,lon)).km

sum_data = []
avail_data = []
cluster_center = []

for nm in area_coverage["employee_name"].dropna().unique():
    cov = area_coverage[area_coverage["employee_name"] == nm]
    if cov.empty:
        continue
    dealers_in = df_dealer[(df_dealer["latitude"].between(cov.min_lat.values[0], cov.max_lat.values[0])) & (df_dealer["longitude"].between(cov.min_lon.values[0], cov.max_lon.values[0]))]
    s = summary[summary["employee_name"] == nm][["date","client_name","lat","lon"]].rename(columns={"lat":"latitude","lon":"longitude"}).copy()
    s["sales_name"] = nm
    a = dealers_in[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].copy()
    a["tag"] = "avail"
    a["sales_name"] = nm
    if len(s) >= 2:
        wcss = []
        for k in range(4, min(9, len(s))):
            X = list(zip(s["latitude"], s["longitude"]))
            km = KMeans(n_clusters=k, n_init="auto").fit(X)
            wcss.append(km.inertia_)
        knee = KneeLocator(range(4, min(9, len(s))), wcss, curve="convex", direction="decreasing")
        n_clusters = knee.elbow if knee.elbow is not None else min(4, len(s))
        n_clusters = max(1, n_clusters)
        km = KMeans(n_clusters=n_clusters, n_init="auto")
        km.fit(list(zip(s["latitude"], s["longitude"])))
        s["cluster"] = km.labels_
        for i in range(len(km.cluster_centers_)):
            a[f"dist_center_{i}"] = a.apply(lambda x: get_distance_dealer(i, x.latitude, x.longitude, km), axis=1)
        cl = pd.DataFrame(km.cluster_centers_, columns=["latitude","longitude"])
    else:
        s["cluster"] = 0
        cl = pd.DataFrame([[s["latitude"].mean() if not s.empty else 0, s["longitude"].mean() if not s.empty else 0]], columns=["latitude","longitude"])
    cl["sales_name"] = nm
    cl["cluster"] = range(len(cl))
    cluster_center.append(cl)
    avail_data.append(a)
    sum_data.append(s)

sum_df = pd.concat(sum_data, ignore_index=True) if sum_data else pd.DataFrame(columns=["date","client_name","latitude","longitude","sales_name","cluster"])
avail_df = pd.concat(avail_data, ignore_index=True) if avail_data else pd.DataFrame(columns=["id_dealer_outlet","brand","business_type","city","name","latitude","longitude","tag","sales_name"])
clust_df = pd.concat(cluster_center, ignore_index=True) if cluster_center else pd.DataFrame(columns=["latitude","longitude","sales_name","cluster"])

grouped_run_order = clean_running_order(running_order)

dist_cols = [c for c in avail_df.columns if c.startswith("dist_center_")]
if dist_cols:
    mv = avail_df[dist_cols].apply(pd.to_numeric, errors="coerce").fillna(1e9).min(axis=1)
    for c in dist_cols:
        col = pd.to_numeric(avail_df[c], errors="coerce")
        avail_df[c] = np.where(col.eq(mv), col, np.nan)

avail_df_merge = avail_df.merge(grouped_run_order, how="left", on="id_dealer_outlet")
ld = location_detail.rename(columns={"City":"city","Cluster":"cluster"})
avail_df_merge = avail_df_merge.merge(ld[["city","cluster"]], how="left", on="city")

cl_map = cluster_left.replace({"CHERY":"Chery","Kia":"KIA"})
cl_map = cl_map.rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability","Category":"Category"})
cl_map = cl_map[cl_map["Category"].astype(str).str.strip().str.title() == "Car"]
avail_df_merge = avail_df_merge.merge(cl_map[["cluster","brand","daily_gen","daily_need","delta","availability"]], how="left", on=["brand","cluster"])

avail_df_merge["nearest_end_date"] = pd.to_datetime(avail_df_merge["nearest_end_date"], errors="coerce")
avail_df_merge["tag"] = np.where(avail_df_merge["nearest_end_date"].isna(), "Not Active", "Active")
