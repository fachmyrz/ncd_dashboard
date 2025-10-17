import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt
from datetime import datetime
from typing import Dict
from data_load import cluster_left, location_detail, df_dealer, df_visit, running_order, sales_orders

pd.options.mode.chained_assignment = None

def haversine_vec(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.minimum(1, np.sqrt(a)))
    km = 6371.0088 * c
    return km

def parse_latlon_field(v):
    if pd.isna(v):
        return (np.nan, np.nan)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return float(v[0]), float(v[1])
        except:
            return (np.nan, np.nan)
    s = str(v).strip()
    s = s.replace("(", "").replace(")", "").replace("'", "").replace('"', "")
    if "," in s:
        a, b = s.split(",", 1)
        try:
            return float(a.strip()), float(b.strip())
        except:
            try:
                return float(b.strip()), float(a.strip())
            except:
                return (np.nan, np.nan)
    return (np.nan, np.nan)

def clean_dealers(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    cols_map = {c.lower().strip(): c for c in df.columns}
    def get(col_names):
        for c in col_names:
            if c in cols_map:
                return df[cols_map[c]]
        return pd.Series(np.nan, index=df.index)
    id_col = get(["id_dealer_outlet", "dealer_id", "id"])
    brand = get(["brand"])
    btype = get(["business_type", "business"])
    city = get(["city", "kota"])
    name = get(["name", "dealer_name", "nama"])
    lat = get(["latitude", "lat"])
    lon = get(["longitude", "long", "lng"])
    out = pd.DataFrame({
        "id_dealer_outlet": id_col.astype(str).str.replace(r"\.0$", "", regex=True).fillna(""),
        "brand": brand.fillna(""),
        "business_type": btype.fillna(""),
        "city": city.fillna(""),
        "name": name.fillna(""),
        "latitude": lat.astype(str).str.replace("`", "").str.strip().replace("", np.nan),
        "longitude": lon.astype(str).str.replace("`", "").str.strip().replace("", np.nan)
    })
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out[out["business_type"].astype(str).str.contains("Car", case=False, na=False)]
    out = out.dropna(subset=["id_dealer_outlet"]).reset_index(drop=True)
    out["id_dealer_outlet"] = out["id_dealer_outlet"].astype(str)
    return out

def clean_visits(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    cols = {c.lower().strip(): c for c in df.columns}
    def g(names):
        for n in names:
            if n in cols:
                return df[cols[n]]
        return pd.Series(np.nan, index=df.index)
    employee = g(["employee name","nama karyawan","nama karyawan ","nama karyawan"])
    client = g(["client name","nama klien","client"])
    datetime_col = g(["date time start","tanggal datang","date","tanggal"])
    latlon_col = g(["latitude & longitude datang","latitude & longitude","latlong","latitude_longitude","latitude & longitude"])
    nik_col = g(["nomor induk karyawan","nik","nomor induk"])
    divisi_col = g(["divisi","division"])
    df2 = pd.DataFrame({
        "employee_name": employee.fillna(""),
        "client_name": client.fillna(""),
        "visit_raw_datetime": datetime_col.fillna(""),
        "latlon_raw": latlon_col.fillna(""),
        "nik": nik_col.fillna(""),
        "divisi": divisi_col.fillna("")
    })
    df2["visit_datetime"] = pd.to_datetime(df2["visit_raw_datetime"].astype(str).str.slice(0,19), errors="coerce")
    parsed = df2["latlon_raw"].apply(parse_latlon_field).tolist()
    ll = pd.DataFrame(parsed, columns=["lat","long"], index=df2.index)
    df2["lat"] = pd.to_numeric(ll["lat"], errors="coerce")
    df2["long"] = pd.to_numeric(ll["long"], errors="coerce")
    df2 = df2[~df2["employee_name"].astype(str).str.lower().isin(["","nan"])].reset_index(drop=True)
    df2 = df2[~df2["nik"].astype(str).str.contains("deleted-", case=False, na=False)]
    df2 = df2[~df2["divisi"].astype(str).str.contains("Trainer", case=False, na=False)]
    return df2

def assign_visits_to_dealers(visits: pd.DataFrame, dealers: pd.DataFrame, max_km: float = 1.0) -> pd.DataFrame:
    if visits is None or visits.empty:
        return pd.DataFrame()
    if dealers is None or dealers.empty:
        v = visits.copy().reset_index(drop=True)
        v["matched_dealer_id"] = np.nan
        v["matched_dealer_name"] = np.nan
        v["distance_km"] = np.nan
        return v
    v = visits.copy().reset_index(drop=True)
    d = dealers.copy().reset_index(drop=True)
    dz = d[["id_dealer_outlet","name","latitude","longitude","city","brand"]].dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    if dz.empty:
        v["matched_dealer_id"] = np.nan
        v["matched_dealer_name"] = np.nan
        v["distance_km"] = np.nan
        return v
    visit_lat = v["lat"].to_numpy(dtype=float)
    visit_lon = v["long"].to_numpy(dtype=float)
    dealer_lat = dz["latitude"].to_numpy(dtype=float)
    dealer_lon = dz["longitude"].to_numpy(dtype=float)
    res_matched = []
    res_name = []
    res_dist = []
    for i in range(len(visit_lat)):
        lat = visit_lat[i]
        lon = visit_lon[i]
        if np.isnan(lat) or np.isnan(lon):
            res_matched.append(np.nan)
            res_name.append(np.nan)
            res_dist.append(np.nan)
            continue
        dists = haversine_vec(lat, lon, dealer_lat, dealer_lon)
        idx = int(np.nanargmin(dists))
        mind = float(dists[idx])
        if np.isfinite(mind) and mind <= max_km:
            res_matched.append(dz.iloc[idx]["id_dealer_outlet"])
            res_name.append(dz.iloc[idx]["name"])
            res_dist.append(mind)
        else:
            res_matched.append(np.nan)
            res_name.append(np.nan)
            res_dist.append(np.nan)
    v["matched_dealer_id"] = res_matched
    v["matched_dealer_name"] = res_name
    v["distance_km"] = res_dist
    return v

def get_summary_data(visits: pd.DataFrame):
    if visits is None or visits.empty:
        return pd.DataFrame(), pd.DataFrame()
    v = visits.copy()
    v["date"] = pd.to_datetime(v["visit_datetime"], errors="coerce").dt.date
    v = v.dropna(subset=["date"]).reset_index(drop=True)
    # --- FIX: use .dt on a datetime64 series
    v["month_year"] = pd.to_datetime(v["date"]).dt.to_period("M").astype(str)
    agg = v.groupby(["month_year","employee_name"], as_index=False).agg(ctd_visit=("client_name","count"))
    # safe compute avg distance
    try:
        avg_dist = v.groupby(["month_year","employee_name"])["distance_km"].mean().reset_index().rename(columns={"distance_km":"avg_distance_km"})
        agg = agg.merge(avg_dist, on=["month_year","employee_name"], how="left")
    except:
        agg["avg_distance_km"] = 0
    agg["avg_time_between_minute"] = 0
    agg["avg_speed_kmpm"] = 0
    return v, agg

def prepare_run_order(running_order_raw: pd.DataFrame) -> pd.DataFrame:
    if running_order_raw is None or running_order_raw.empty:
        return pd.DataFrame(columns=["id_dealer_outlet","joined_dse","active_dse","nearest_end_date"])
    ro = running_order_raw.copy()
    cols = {c.lower().strip(): c for c in ro.columns}
    def g(names):
        for n in names:
            if n in cols:
                return ro[cols[n]]
        return pd.Series(np.nan, index=ro.index)
    idc = g(["dealer id","dealer_id","dealer"])
    joined = g(["lms id","joined_dse","lms_id","lms"])
    active = g(["isactive","active_dse","is_active","active"])
    end = g(["end date","nearest_end_date","end_date","end"])
    out = pd.DataFrame({
        "id_dealer_outlet": idc.astype(str).str.replace(r"\.0$", "", regex=True).fillna(""),
        "joined_dse": joined.fillna("0"),
        "active_dse": pd.to_numeric(active, errors="coerce").fillna(0).astype(int),
        "nearest_end_date": pd.to_datetime(end, errors="coerce")
    })
    grouped = out.groupby("id_dealer_outlet", as_index=False).agg(joined_dse=("joined_dse","count"), active_dse=("active_dse","sum"), nearest_end_date=("nearest_end_date","min"))
    return grouped

def compute_availability(avail_df: pd.DataFrame, run_order_grouped: pd.DataFrame, location_df: pd.DataFrame, cluster_left_df: pd.DataFrame) -> pd.DataFrame:
    if avail_df is None or avail_df.empty:
        return pd.DataFrame()
    a = avail_df.copy()
    if "id_dealer_outlet" not in a.columns:
        # try finding possible id column
        possible = [c for c in a.columns if "id" in c.lower() and "dealer" in c.lower()]
        if possible:
            a = a.rename(columns={possible[0]:"id_dealer_outlet"})
        else:
            a["id_dealer_outlet"] = a.index.astype(str)
    a["id_dealer_outlet"] = a["id_dealer_outlet"].astype(str)
    ro = run_order_grouped.copy() if run_order_grouped is not None else pd.DataFrame()
    if not ro.empty and "id_dealer_outlet" in ro.columns:
        a = a.merge(ro, how="left", on="id_dealer_outlet")
    # merge cluster info using location_df
    ld = location_df.copy() if location_df is not None else pd.DataFrame()
    if {"City","Cluster"}.issubset(set(ld.columns)):
        ld2 = ld.rename(columns={"City":"city","Cluster":"cluster"})
        a = a.merge(ld2[["city","cluster"]], how="left", on="city")
    if cluster_left_df is not None and not cluster_left_df.empty:
        cl = cluster_left_df.copy()
        # try useful columns and merge on brand+cluster if present
        if "Cluster" in cl.columns:
            cl = cl.rename(columns={c:c for c in cl.columns})
            col_map = {c: c for c in cl.columns}
            cl2 = cl.rename(columns={"Cluster":"cluster","Brand":"brand","Delta":"delta","Tag":"availability"})
            merge_cols = [c for c in ["brand","cluster"] if c in a.columns and c in cl2.columns]
            if merge_cols:
                a = a.merge(cl2, how="left", on=merge_cols)
    a["tag"] = np.where(a.get("nearest_end_date").isna(), "Not Active", "Active")
    return a

def compute_cluster_centers(avail_df_merge: pd.DataFrame) -> pd.DataFrame:
    if avail_df_merge is None or avail_df_merge.empty:
        return pd.DataFrame()
    if "cluster" in avail_df_merge.columns:
        centers = avail_df_merge.groupby("cluster").agg(longitude=("longitude","mean"), latitude=("latitude","mean"), count_dealers=("id_dealer_outlet","nunique")).reset_index()
    else:
        centers = pd.DataFrame()
    return centers

def compute_all(sheets: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    sheets = sheets or {}
    dealers_raw = sheets.get("dealers") if sheets.get("dealers") is not None else df_dealer
    visits_raw = sheets.get("visits") if sheets.get("visits") is not None else df_visit
    location_df = sheets.get("location") if sheets.get("location") is not None else location_detail
    cluster_left_df = sheets.get("need_cluster") if sheets.get("need_cluster") is not None else cluster_left
    running_order_raw = sheets.get("running_order") if sheets.get("running_order") is not None else running_order
    orders_raw = sheets.get("orders") if sheets.get("orders") is not None else sales_orders
    dealers = clean_dealers(dealers_raw)
    visits = clean_visits(visits_raw)
    visits = visits.reset_index(drop=True)
    visits = assign_visits_to_dealers(visits, dealers, max_km=1.0)
    summary_visits, summary_agg = get_summary_data(visits)
    ro_group = prepare_run_order(running_order_raw)
    avail_df_merge = compute_availability(dealers, ro_group, location_df, cluster_left_df)
    # ensure numeric lat/lon
    if "latitude" in avail_df_merge.columns:
        avail_df_merge["latitude"] = pd.to_numeric(avail_df_merge["latitude"], errors="coerce")
    if "longitude" in avail_df_merge.columns:
        avail_df_merge["longitude"] = pd.to_numeric(avail_df_merge["longitude"], errors="coerce")
    clust_df = compute_cluster_centers(avail_df_merge)
    revenue_monthly = pd.DataFrame()
    if orders_raw is not None and not orders_raw.empty:
        ords = orders_raw.copy()
        possible_date_cols = [c for c in ords.columns if "order_date" in c.lower() or c.lower()=="date" or "date" in c.lower()]
        datecol = possible_date_cols[0] if possible_date_cols else None
        if datecol:
            ords["order_date"] = pd.to_datetime(ords[datecol], errors="coerce")
            ords["month_year"] = ords["order_date"].dt.to_period("M").astype(str)
            amtcols = [c for c in ords.columns if "total_paid_after_tax" in c.lower() or "amount" in c.lower() or "total" in c.lower()]
            if amtcols:
                a = amtcols[0]
                ords[a] = pd.to_numeric(ords[a], errors="coerce").fillna(0)
                idcols = [c for c in ords.columns if "dealer_id" in c.lower() or "id_dealer" in c.lower()]
                idcol = idcols[0] if idcols else None
                if idcol:
                    ords["id_dealer_outlet"] = ords[idcol].astype(str)
                    revenue_monthly = ords.groupby(["month_year","id_dealer_outlet"], as_index=False)[a].sum().rename(columns={a:"monthly_revenue"})
    return {"sum_df": summary_agg, "clust_df": clust_df, "avail_df_merge": avail_df_merge, "df_visits": visits, "revenue_monthly": revenue_monthly}
