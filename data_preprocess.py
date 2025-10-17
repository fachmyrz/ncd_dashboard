# data_preprocess.py
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import data_load

EARTH_R = 6371.0088  # km

# --- helpers ---
def _to_float_safe(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace('`','').replace('\u200b','')
    # allow minus and decimal
    m = re.search(r'-?\d+(\.\d+)?', s)
    return float(m.group(0)) if m else np.nan

def _parse_latlon_cell(val):
    if pd.isna(val): return (np.nan, np.nan)
    s = str(val)
    if ',' in s:
        a,b = s.split(',',1)
        return (_to_float_safe(a), _to_float_safe(b))
    nums = re.findall(r'-?\d+\.\d+|-?\d+', s)
    if len(nums) >= 2:
        return (float(nums[0]), float(nums[1]))
    return (np.nan, np.nan)

# --- cleaners ---
def clean_dealers(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    # standardize column names (lowercase)
    df.columns = [c.strip() for c in df.columns]
    # map common names
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("id_dealer_outlet","dealer_id","id"):
            mapping[c] = "id_dealer_outlet"
        if lc in ("name","dealer name","dealer_name"):
            mapping[c] = "name"
        if lc == "brand":
            mapping[c] = "brand"
        if lc in ("business_type","type"):
            mapping[c] = "business_type"
        if lc == "city":
            mapping[c] = "city"
        if lc in ("latitude","lat"):
            mapping[c] = "latitude"
        if lc in ("longitude","lon","long"):
            mapping[c] = "longitude"
    df = df.rename(columns=mapping)
    # only cars
    if "business_type" in df.columns:
        df = df[df["business_type"].astype(str).str.lower().str.contains("car", na=False)]
    # parse lat/lon
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].apply(_to_float_safe)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].apply(_to_float_safe)
    if "id_dealer_outlet" in df.columns:
        df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce")
    df = df.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    return df

def clean_visits(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    # map columns we expect
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("nama karyawan","employee name","employee_name","nama_karyawan"):
            mapping[c] = "employee_name"
        if lc in ("nama klien","client name","client_name"):
            mapping[c] = "client_name"
        if lc in ("tanggal datang","date time start","date_time_start"):
            mapping[c] = "visit_datetime"
        if "latitude" in lc and "longitude" in lc:
            mapping[c] = "latlong"
        if lc in ("nomor induk karyawan","nik","nomor_induk_karyawan"):
            mapping[c] = "nik"
        if lc == "divisi":
            mapping[c] = "divisi"
    df = df.rename(columns=mapping)
    # datetime
    if "visit_datetime" in df.columns:
        df["visit_datetime"] = pd.to_datetime(df["visit_datetime"].astype(str), errors="coerce")
    # latlong parse
    if "latlong" in df.columns:
        latlon = df["latlong"].apply(_parse_latlon_cell)
        df["lat"] = latlon.apply(lambda t: t[0])
        df["long"] = latlon.apply(lambda t: t[1])
    else:
        # fallback separate columns
        for cand_lat in ("latitude","lat"):
            if cand_lat in df.columns:
                df["lat"] = df[cand_lat].apply(_to_float_safe)
                break
        for cand_lon in ("longitude","long","lon"):
            if cand_lon in df.columns:
                df["long"] = df[cand_lon].apply(_to_float_safe)
                break
    # clean nik and divisi
    if "nik" in df.columns:
        df = df[~df["nik"].astype(str).str.contains("deleted-", na=False)]
    if "divisi" in df.columns:
        df = df[~df["divisi"].astype(str).str.contains("trainer", case=False, na=False)]
    # ensure employee_name present
    if "employee_name" in df.columns:
        df["employee_name"] = df["employee_name"].astype(str).str.strip()
        df = df[~df["employee_name"].isna()].reset_index(drop=True)
    return df

# --- matching using BallTree (fast) ---
def assign_visits_to_dealers(visits: pd.DataFrame, dealers: pd.DataFrame, max_km=1.0) -> pd.DataFrame:
    """
    Vectorized assignment:
    - first perform exact case-insensitive client_name match
    - for unmatched visits with valid lat/lon, use BallTree (haversine metric) to find nearest dealers within max_km
    """
    if visits is None or visits.empty:
        return pd.DataFrame()
    v = visits.copy().reset_index(drop=True)
    d = dealers.copy().reset_index(drop=True)
    # prepare dealer coords (radians) for BallTree
    d_coords = d[["latitude","longitude"]].to_numpy(dtype=float)
    valid_dealer_mask = np.isfinite(d_coords).all(axis=1)
    if valid_dealer_mask.sum() == 0:
        v["matched_name"] = np.nan
        v["matched_dealer_id"] = np.nan
        return v
    d_valid = d.loc[valid_dealer_mask].reset_index(drop=True)
    d_rad = np.radians(d_valid[["latitude","longitude"]].to_numpy(dtype=float))
    tree = BallTree(d_rad, metric="haversine")

    # name -> dealer mapping (lowercase)
    name_to_row = {}
    if "name" in d.columns:
        for i, nm in d["name"].dropna().astype(str).str.strip().iteritems():
            name_to_row[nm.strip().lower()] = i

    matched_name = []
    matched_id = []

    # prepare visit coords in bulk
    lat_arr = v.get("lat", pd.Series(np.nan)).apply(_to_float_safe).to_numpy(dtype=float)
    lon_arr = v.get("long", pd.Series(np.nan)).apply(_to_float_safe).to_numpy(dtype=float)
    has_coords = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    # first: exact name match (fast)
    for idx, row in v.iterrows():
        cname = str(row.get("client_name","")).strip()
        if cname:
            key = cname.lower()
            if key in name_to_row:
                # exact match found
                di = name_to_row[key]
                matched_name.append(d.loc[di, "name"])
                matched_id.append(d.loc[di, "id_dealer_outlet"] if "id_dealer_outlet" in d.columns else np.nan)
                continue
        # placeholder for now, will fill using spatial matching
        matched_name.append(np.nan)
        matched_id.append(np.nan)

    # spatial match for those with no name match and have coords
    unmatched_idxs = [i for i,x in enumerate(matched_name) if pd.isna(x) and has_coords[i]]
    if len(unmatched_idxs) > 0:
        pts = np.vstack([lat_arr[unmatched_idxs], lon_arr[unmatched_idxs]]).T
        pts_rad = np.radians(pts)
        # query radius in radians
        rad = max_km / EARTH_R
        # query the tree
        ind_arrays = tree.query_radius(pts_rad, r=rad, return_distance=False)
        for pos, inds in zip(unmatched_idxs, ind_arrays):
            if len(inds) == 0:
                continue
            # if multiple, pick the closest by distance (compute distances)
            cand_coords = d_rad[inds]
            pt = pts_rad[np.where(np.array(unmatched_idxs)==pos)[0][0]]
            # compute great-circle distances vectorized
            dists = haversine_vec_rad(pt, cand_coords)  # we'll define this small helper below
            best = inds[np.argmin(dists)]
            matched_name[pos] = d_valid.loc[best, "name"]
            matched_id[pos] = d_valid.loc[best, "id_dealer_outlet"] if "id_dealer_outlet" in d_valid.columns else np.nan

    v["matched_name"] = matched_name
    v["matched_dealer_id"] = matched_id
    return v

def haversine_vec_rad(pt_rad, arr_rad):
    """
    pt_rad: (2,) lat/lon in radians
    arr_rad: (N,2) lat/lon in radians
    returns distances in km
    """
    lat1 = pt_rad[0]; lon1 = pt_rad[1]
    lat2 = arr_rad[:,0]; lon2 = arr_rad[:,1]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    return EARTH_R * c

# --- orchestrator ---
def compute_all():
    """Load, clean, match and return processed frames."""
    sheets = data_load.get_sheets()
    dealers_raw = sheets.get("df_dealer", pd.DataFrame())
    visits_raw = sheets.get("df_visit", pd.DataFrame())
    location_detail = sheets.get("location_detail", pd.DataFrame())
    running_order = sheets.get("running_order", pd.DataFrame())

    dealers = clean_dealers(dealers_raw)
    visits = clean_visits(visits_raw)
    visits = assign_visits_to_dealers(visits, dealers, max_km=1.0)

    # prepare run order (joined/active) quick map
    ro = running_order.copy() if running_order is not None else pd.DataFrame()
    ro_cols = {c: c for c in ro.columns}
    # attempt common names mapping safely:
    if not ro.empty:
        ro.columns = [c.strip() for c in ro.columns]
        # try find id/joined/active/enddate columns
        idcol = next((c for c in ro.columns if c.lower() in ("id_dealer_outlet","dealer_id","id")), None)
        jcol = next((c for c in ro.columns if "joined" in c.lower()), None)
        acol = next((c for c in ro.columns if "active" in c.lower()), None)
        ecol = next((c for c in ro.columns if "end" in c.lower() or "date" in c.lower()), None)
        ren = {}
        if idcol: ren[idcol] = "id_dealer_outlet"
        if jcol: ren[jcol] = "joined_dse"
        if acol: ren[acol] = "active_dse"
        if ecol: ren[ecol] = "nearest_end_date"
        ro = ro.rename(columns=ren)
        if "id_dealer_outlet" in ro.columns:
            ro["id_dealer_outlet"] = pd.to_numeric(ro["id_dealer_outlet"], errors="coerce")
            ro["joined_dse"] = pd.to_numeric(ro.get("joined_dse", 0), errors="coerce").fillna(0).astype(int)
            ro["active_dse"] = pd.to_numeric(ro.get("active_dse", 0), errors="coerce").fillna(0).astype(int)
            if "nearest_end_date" in ro.columns:
                ro["nearest_end_date"] = pd.to_datetime(ro["nearest_end_date"], errors="coerce")
            ro_group = ro.groupby("id_dealer_outlet", dropna=True).agg({"joined_dse":"sum","active_dse":"sum","nearest_end_date":"min"}).reset_index()
        else:
            ro_group = pd.DataFrame()
    else:
        ro_group = pd.DataFrame()

    # compute availability merge
    avail = dealers.copy()
    if not ro_group.empty and "id_dealer_outlet" in avail.columns:
        try:
            avail = avail.merge(ro_group, how="left", on="id_dealer_outlet")
        except Exception:
            pass

    # merge city->cluster if location_detail present
    if location_detail is not None and not location_detail.empty:
        cols = [c.strip() for c in location_detail.columns]
        citycol = next((c for c in cols if c.lower() == "city" or "city" in c.lower()), None)
        clustcol = next((c for c in cols if c.lower() in ("cluster","area")), None)
        if citycol and clustcol:
            ld = location_detail.rename(columns={citycol:"city", clustcol:"cluster"})
            try:
                avail = avail.merge(ld[["city","cluster"]], how="left", on="city")
            except Exception:
                pass

    # simple cluster centers for visualization from visits grouped by employee_name (centroid only)
    centers = []
    if not visits.empty:
        grp = visits.dropna(subset=["lat","long"]).groupby("employee_name")
        for name, g in grp:
            lat = g["lat"].astype(float).mean()
            lon = g["long"].astype(float).mean()
            centers.append({"sales_name": name, "cluster": "center", "latitude": lat, "longitude": lon})
    clust_df = pd.DataFrame(centers)

    # order areas by dealer count (for UI)
    area_order = []
    if not location_detail.empty and "city" in location_detail.columns and "Cluster" in location_detail.columns:
        ld = location_detail.rename(columns={c:c for c in location_detail.columns})
        if "City" in ld.columns and "Cluster" in ld.columns:
            tmp = dealers.merge(ld[["City","Cluster"]].rename(columns={"City":"city","Cluster":"cluster"}), how="left", on="city")
            area_order = tmp.groupby("cluster").size().reset_index(name="ct").sort_values("ct", ascending=False)["cluster"].dropna().astype(str).tolist()

    return {
        "dealers": dealers,
        "visits": visits,
        "avail": avail,
        "clust_df": clust_df,
        "area_order": area_order
    }
