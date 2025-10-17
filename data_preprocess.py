# data_preprocess.py
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import data_load   # uses st.secrets inside data_load; fine

# ---------- helpers ----------
def _first_col(df, candidates, default=None):
    if df is None or df.empty:
        return default
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand and cand.lower() in cols:
            return cols[cand.lower()]
    return default

def _to_float_safe(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # remove stray backticks and weird chars
    s = s.replace('`','').replace('\u200b','')
    # allow decimal and minus
    s = re.sub(r'[^0-9\.\-]', ' ', s).strip()
    # pick first numeric token
    m = re.search(r'-?\d+(\.\d+)?', s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except:
        return np.nan

def haversine_np(lat1, lon1, lat2, lon2):
    # accepts scalars or numpy arrays
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0088 * c

def _parse_latlon_field(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip()
    # typical format "-6.9486862, 107.62317"
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            return (_to_float_safe(parts[0]), _to_float_safe(parts[1]))
    # fallback: find first two floats
    nums = re.findall(r'-?\d+\.\d+|-?\d+', s)
    if len(nums) >= 2:
        return (_to_float_safe(nums[0]), _to_float_safe(nums[1]))
    return (np.nan, np.nan)

# ---------- cleaners ----------
def clean_dealers(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # detect columns (fall back to common names)
    id_col = _first_col(df, ["id_dealer_outlet","Dealer Id","dealer_id","id"])
    name_col = _first_col(df, ["name","Dealer Name","dealer_name"])
    brand_col = _first_col(df, ["brand","Brand"])
    bt_col = _first_col(df, ["business_type","Business Type","type"])
    city_col = _first_col(df, ["city","City","kota"])
    lat_col = _first_col(df, ["latitude","Latitude","lat"])
    lon_col = _first_col(df, ["longitude","Longitude","long","lon"])
    remap = {}
    if id_col: remap[id_col] = "id_dealer_outlet"
    if name_col: remap[name_col] = "name"
    if brand_col: remap[brand_col] = "brand"
    if bt_col: remap[bt_col] = "business_type"
    if city_col: remap[city_col] = "city"
    if lat_col: remap[lat_col] = "latitude"
    if lon_col: remap[lon_col] = "longitude"
    df = df.rename(columns=remap)
    # keep only Car dealers (robust)
    if "business_type" in df.columns:
        df = df[df["business_type"].astype(str).str.lower().str.contains("car", na=False)]
    # safe float conversion for lat/lon
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].apply(_to_float_safe)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].apply(_to_float_safe)
    if "id_dealer_outlet" in df.columns:
        df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce")
    df = df.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    return df

def clean_visits(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    col_employee = _first_col(df, ["Nama Karyawan","Employee Name","employee_name","Nama Karyawan"])
    col_client = _first_col(df, ["Nama Klien","Client Name","client_name","Client"])
    col_date = _first_col(df, ["Tanggal Datang","Date Time Start","date_time_start","Tanggal"])
    col_latlon = _first_col(df, ["Latitude & Longitude Datang","Latitude & Longitude","latlong","Latitude & Longitude","LatitudeLongitude"])
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
    # parse datetime
    if "visit_datetime" in df.columns:
        df["visit_datetime"] = pd.to_datetime(df["visit_datetime"].astype(str), errors="coerce")
    # parse lat/long single column
    if "latlong" in df.columns:
        parsed = df["latlong"].apply(_parse_latlon_field)
        df["lat"] = parsed.apply(lambda t: t[0])
        df["long"] = parsed.apply(lambda t: t[1])
    else:
        latc = _first_col(df, ["Latitude","latitude","Lat"])
        lonc = _first_col(df, ["Longitude","longitude","Lon","long"])
        if latc and lonc:
            df["lat"] = df[latc].apply(_to_float_safe)
            df["long"] = df[lonc].apply(_to_float_safe)
    # exclude deleted- in nik and Divisi trainers
    if "nik" in df.columns:
        df["nik"] = df["nik"].astype(str)
        df = df[~df["nik"].str.contains("deleted-", na=False)]
    if "divisi" in df.columns:
        df = df[~df["divisi"].astype(str).str.contains("trainer", case=False, na=False)]
    # ensure employee_name exists
    if "employee_name" in df.columns:
        df["employee_name"] = df["employee_name"].astype(str).str.strip()
        df = df.dropna(subset=["employee_name"]).reset_index(drop=True)
    return df

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    if visits is None:
        return pd.DataFrame()
    v = visits.copy().reset_index(drop=True)
    if dealers is None or dealers.empty:
        v["matched_name"] = np.nan
        v["matched_dealer_id"] = np.nan
        return v
    d = dealers.copy().reset_index(drop=True)
    d["latitude"] = d["latitude"].apply(_to_float_safe)
    d["longitude"] = d["longitude"].apply(_to_float_safe)
    dcoords = d[["id_dealer_outlet","name","latitude","longitude"]].dropna().reset_index(drop=True)
    if dcoords.empty:
        v["matched_name"] = np.nan
        v["matched_dealer_id"] = np.nan
        return v
    dlat = dcoords["latitude"].to_numpy(dtype=float)
    dlon = dcoords["longitude"].to_numpy(dtype=float)

    matched_name = []
    matched_id = []
    for _, row in v.iterrows():
        lat = _to_float_safe(row.get("lat", np.nan))
        lon = _to_float_safe(row.get("long", np.nan))
        cname = str(row.get("client_name","")).strip()
        # exact name match first (case-insensitive)
        if cname and cname.lower() != "nan":
            cand = dcoords[dcoords["name"].astype(str).str.strip().str.lower() == cname.lower()]
            if len(cand) == 1:
                matched_name.append(cand.iloc[0]["name"])
                matched_id.append(cand.iloc[0]["id_dealer_outlet"])
                continue
        if np.isnan(lat) or np.isnan(lon):
            matched_name.append(np.nan)
            matched_id.append(np.nan)
            continue
        lat_arr = np.full_like(dlat, lat, dtype=float)
        lon_arr = np.full_like(dlon, lon, dtype=float)
        dists = haversine_np(lat_arr, lon_arr, dlat, dlon)
        idx = int(np.nanargmin(dists))
        km = float(dists[idx])
        if km <= max_km:
            matched_name.append(dcoords.iloc[idx]["name"])
            matched_id.append(dcoords.iloc[idx]["id_dealer_outlet"])
        else:
            matched_name.append(np.nan)
            matched_id.append(np.nan)
    v["matched_name"] = matched_name
    v["matched_dealer_id"] = matched_id
    return v

def prepare_run_order(running_order):
    if running_order is None or running_order.empty:
        return pd.DataFrame()
    df = running_order.copy()
    id_col = _first_col(df, ["id_dealer_outlet","Dealer Id","dealer_id"])
    joined_col = _first_col(df, ["joined_dse","LMS Id","joined"])
    active_col = _first_col(df, ["active_dse","IsActive","active"])
    enddate_col = _first_col(df, ["nearest_end_date","End Date","EndDate"])
    remap = {}
    if id_col: remap[id_col] = "id_dealer_outlet"
    if joined_col: remap[joined_col] = "joined_dse"
    if active_col: remap[active_col] = "active_dse"
    if enddate_col: remap[enddate_col] = "nearest_end_date"
    df = df.rename(columns=remap)
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
    centers = []
    for name, grp in visits.groupby("employee_name"):
        pts = grp[["lat","long"]].dropna()
        if pts.empty:
            continue
        k = min(4, max(1, int(len(pts)/10)+1)) if len(pts) >= 2 else 1
        try:
            km = KMeans(n_clusters=k, random_state=42).fit(pts)
            labs = km.labels_
            ctrs = km.cluster_centers_
        except Exception:
            labs = np.zeros(len(pts), dtype=int)
            ctrs = np.array([pts.mean().tolist()])
        for i, c in enumerate(ctrs):
            centers.append({"sales_name": name, "cluster": int(i), "latitude": float(c[0]), "longitude": float(c[1])})
    clust_df = pd.DataFrame(centers)
    return pd.DataFrame(), clust_df

def compute_availability(dealers, ro_group, location_detail, clust_df):
    if dealers is None or dealers.empty:
        return pd.DataFrame()
    avail = dealers.copy()
    # compute dist_center_N columns
    if clust_df is not None and not clust_df.empty:
        for i in clust_df["cluster"].unique():
            row = clust_df[clust_df["cluster"] == i].iloc[0]
            lat_c = float(row["latitude"])
            lon_c = float(row["longitude"])
            col = f"dist_center_{i}"
            avail[col] = haversine_np(avail["latitude"].to_numpy(dtype=float), avail["longitude"].to_numpy(dtype=float), np.full(len(avail), lat_c), np.full(len(avail), lon_c))
    # merge run order info
    if ro_group is not None and not ro_group.empty:
        try:
            avail = avail.merge(ro_group, how="left", on="id_dealer_outlet")
        except Exception:
            pass
    # merge area (City->cluster) from location_detail sheet
    if location_detail is not None and not location_detail.empty:
        ld = location_detail.copy()
        city_col = _first_col(ld, ["City","city","City Name"])
        cluster_col = _first_col(ld, ["Cluster","cluster","Area"])
        if city_col and cluster_col:
            ld = ld.rename(columns={city_col:"city", cluster_col:"cluster"})
            try:
                avail = avail.merge(ld[["city","cluster"]], how="left", on="city")
            except Exception:
                pass
    return avail

def get_area_order(location_detail, dealers):
    # return list of areas ordered by dealer count desc
    if location_detail is None or location_detail.empty or dealers is None or dealers.empty:
        return []
    ld = location_detail.copy()
    city_col = _first_col(ld, ["City","city","City Name"])
    cluster_col = _first_col(ld, ["Cluster","cluster","Area"])
    if not city_col or not cluster_col:
        return []
    ld = ld.rename(columns={city_col:"city", cluster_col:"cluster"})
    merged = dealers.merge(ld[["city","cluster"]], how="left", on="city")
    order = merged.groupby("cluster").size().reset_index(name="ct").sort_values("ct", ascending=False)
    return order["cluster"].dropna().astype(str).tolist()

# ---------- orchestrator ----------
def compute_all():
    """
    Load raw sheets via data_load and return dictionary of processed outputs.
    This function is intentionally NOT importing streamlit to avoid circular imports.
    """
    dealers_raw = getattr(data_load, "df_dealer", pd.DataFrame())
    visits_raw = getattr(data_load, "df_visit", pd.DataFrame())
    location_detail = getattr(data_load, "location_detail", pd.DataFrame())
    need_cluster = getattr(data_load, "cluster_left", pd.DataFrame())
    running_order_raw = getattr(data_load, "running_order", pd.DataFrame())

    dealers = clean_dealers(dealers_raw)
    visits = clean_visits(visits_raw)
    visits = assign_visits_to_dealers(visits, dealers, max_km=1.0)
    _, clust_df = compute_clusters(visits)
    ro_group = prepare_run_order(running_order_raw)
    avail_df_merge = compute_availability(dealers, ro_group, location_detail, clust_df)
    areas_ordered = get_area_order(location_detail, dealers)
    return {
        "dealers": dealers,
        "visits": visits,
        "clust_df": clust_df,
        "avail_df_merge": avail_df_merge,
        "areas_ordered": areas_ordered
    }
