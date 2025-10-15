import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from difflib import get_close_matches
from data_load import df_dealer, df_visit, sales_data, running_order, location_detail, cluster_left

def clean_dealers(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["dealer id","dealer_id","id_dealer_outlet","id"]:
            colmap[c] = "id_dealer_outlet"
        if "client" in cl and "name" in cl or "nama klien" in cl:
            colmap[c] = "client_name"
        if cl in ["name","dealer name","outlet","outlet name","nama dealer"]:
            colmap[c] = "name"
        if "brand" in cl:
            colmap[c] = "brand"
        if "city" in cl:
            colmap[c] = "city"
        if "latitude" in cl or cl == "lat":
            colmap[c] = "latitude"
        if "longitude" in cl or cl == "long" or cl == "lon":
            colmap[c] = "longitude"
        if "total_dse" in cl or ("total" in cl and "dse" in cl):
            colmap[c] = "total_dse"
        if "business" in cl and "type" in cl:
            colmap[c] = "business_type"
    df = df.rename(columns=colmap)
    if "id_dealer_outlet" in df.columns:
        df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce")
    for k in ["latitude","longitude"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k].astype(str).str.replace(",","").str.strip("."), errors="coerce")
    if "total_dse" not in df.columns:
        df["total_dse"] = 0
    else:
        df["total_dse"] = pd.to_numeric(df["total_dse"], errors="coerce").fillna(0).astype(int)
    df["client_name"] = df.get("client_name", df.get("name", pd.Series([""]*len(df)))).astype(str).str.strip()
    if "business_type" in df.columns:
        df["business_type"] = df["business_type"].astype(str).str.title()
    return df

def _parse_combined_ll(s):
    try:
        s = str(s).strip()
        s = s.replace("(", "").replace(")", "")
        parts = [p.strip() for p in s.split(",") if p.strip()!=""]
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
    except:
        pass
    return np.nan, np.nan

def clean_visits(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    div_col = None
    for c in df.columns:
        cl = c.lower()
        if "tanggal" in cl or "date" in cl or "visit" in cl:
            if "time" in cl or "datetime" in cl:
                colmap[c] = "visit_datetime"
            else:
                colmap[c] = "visit_datetime"
        if "nama klien" in cl or ("client" in cl and "name" in cl) or "nama_klien" in cl:
            colmap[c] = "client_name"
        if "nama karyawan" in cl or ("employee" in cl and "name" in cl) or "bde" in cl:
            colmap[c] = "employee_name"
        if "nomor" in cl or "nik" in cl or "employee_id" in cl:
            colmap[c] = "employee_id"
        if "divisi" in cl or "division" in cl:
            div_col = c
    df = df.rename(columns=colmap)
    if "visit_datetime" in df.columns:
        df["visit_datetime"] = pd.to_datetime(df["visit_datetime"], errors="coerce")
    else:
        df["visit_datetime"] = pd.NaT
    if "client_name" in df.columns:
        df["client_name"] = df["client_name"].astype(str).str.strip()
    if "employee_id" in df.columns:
        df = df[~df["employee_id"].astype(str).str.contains("deleted-", na=False)]
    if div_col:
        df = df[~df[div_col].astype(str).str.lower().eq("trainer")]
    lat_col = None
    lon_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ["latitude","lat"] and lat_col is None:
            lat_col = c
        if cl in ["longitude","long","lon"] and lon_col is None:
            lon_col = c
    if lat_col and lon_col:
        df["latitude"] = pd.to_numeric(df[lat_col].astype(str).str.replace(",",""), errors="coerce")
        df["longitude"] = pd.to_numeric(df[lon_col].astype(str).str.replace(",",""), errors="coerce")
    else:
        found = False
        for c in df.columns:
            if "lat" in c.lower() and "long" in c.lower() or "latlong" in c.lower() or ("latitude" in c.lower() and "longitude" in c.lower()):
                parsed = df[c].astype(str).apply(_parse_combined_ll)
                df["latitude"] = pd.to_numeric(pd.Series([t[0] for t in parsed], index=df.index), errors="coerce")
                df["longitude"] = pd.to_numeric(pd.Series([t[1] for t in parsed], index=df.index), errors="coerce")
                found = True
                break
        if not found:
            df["latitude"] = pd.to_numeric(df.get("latitude", pd.Series([np.nan]*len(df))), errors="coerce")
            df["longitude"] = pd.to_numeric(df.get("longitude", pd.Series([np.nan]*len(df))), errors="coerce")
    return df

def clean_orders(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if "dealer" in cl and "id" in cl:
            colmap[c] = "dealer_id"
        if "order_date" in cl or "date" in cl:
            colmap[c] = "order_date"
        if "total_paid_after_tax" in cl or "amount" in cl or "total" in cl:
            colmap[c] = "total_paid_after_tax"
    df = df.rename(columns=colmap)
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "dealer_id" in df.columns:
        df["dealer_id"] = pd.to_numeric(df["dealer_id"], errors="coerce")
    if "total_paid_after_tax" in df.columns:
        df["total_paid_after_tax"] = pd.to_numeric(df["total_paid_after_tax"], errors="coerce").fillna(0.0)
    return df

def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    lat1_r = radians(lat1)
    lon1_r = radians(lon1)
    lat2_r = np.radians(lat2_arr.astype(float))
    lon2_r = np.radians(lon2_arr.astype(float))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df_dealer = clean_dealers(df_dealer) if 'df_dealer' in globals() else pd.DataFrame()
df_visit = clean_visits(df_visit) if 'df_visit' in globals() else pd.DataFrame()
sales_orders = clean_orders(sales_data) if 'sales_data' in globals() else pd.DataFrame()

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    if visits is None or visits.empty:
        return visits
    if dealers is None or dealers.empty:
        visits["client_name_assigned"] = visits.get("client_name", None)
        return visits
    visits = visits.copy()
    dealers_idx = dealers.dropna(subset=["latitude","longitude"])[["id_dealer_outlet","client_name","latitude","longitude"]].reset_index(drop=True)
    dealers_idx["client_lower"] = dealers_idx["client_name"].astype(str).str.lower()
    dealer_names = dealers_idx["client_lower"].tolist()
    dealers_lat = dealers_idx["latitude"].astype(float).to_numpy()
    dealers_lon = dealers_idx["longitude"].astype(float).to_numpy()
    def match_row(row):
        cn = str(row.get("client_name", "")).strip()
        lat = row.get("latitude")
        lon = row.get("longitude")
        if cn:
            cl = cn.lower()
            if cl in dealers_idx["client_lower"].values:
                return dealers_idx.loc[dealers_idx["client_lower"]==cl, "client_name"].iloc[0]
            close = get_close_matches(cl, dealer_names, n=1, cutoff=0.85)
            if close:
                return dealers_idx.loc[dealers_idx["client_lower"]==close[0], "client_name"].iloc[0]
        try:
            lat = float(lat)
            lon = float(lon)
        except:
            return None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        dists = haversine_km(lat, lon, dealers_lat, dealers_lon)
        if np.all(np.isnan(dists)):
            return None
        idx = int(np.nanargmin(dists))
        if dists[idx] <= max_km:
            return dealers_idx.loc[idx, "client_name"]
        return None
    visits["matched_client"] = visits.apply(match_row, axis=1)
    visits["client_name_assigned"] = visits.get("client_name")
    visits["client_name_assigned"] = visits["client_name_assigned"].where(visits["client_name_assigned"].notna() & visits["client_name_assigned"].astype(str).str.strip().ne(""), visits["matched_client"])
    visits["client_name_assigned"] = visits["client_name_assigned"].where(visits["client_name_assigned"].notna(), None)
    return visits

df_visit = assign_visits_to_dealers(df_visit, df_dealer, max_km=1.0)

def compute_visit_metrics(visits, window_days=90):
    if visits is None or visits.empty:
        return pd.DataFrame(columns=["client_name","visits_last_N","last_visit_datetime","last_visited_by","avg_weekly_visits"])
    v = visits.copy()
    v["client_name"] = v.get("client_name_assigned", v.get("client_name"))
    v = v.dropna(subset=["client_name"])
    v = v[~v["client_name"].astype(str).str.strip().eq("")]
    v["visit_datetime"] = pd.to_datetime(v["visit_datetime"], errors="coerce")
    today = pd.to_datetime(datetime.utcnow().date())
    since = today - pd.Timedelta(days=window_days)
    recent = v[v["visit_datetime"] >= since]
    if recent.empty:
        return pd.DataFrame(columns=["client_name","visits_last_N","last_visit_datetime","last_visited_by","avg_weekly_visits"])
    agg = recent.groupby("client_name").agg(visits_last_N=("visit_datetime","count"), last_visit_datetime=("visit_datetime","max")).reset_index()
    if "employee_name" in v.columns:
        idx = v.groupby("client_name")["visit_datetime"].idxmax()
        last_by = v.loc[idx, ["client_name","employee_name"]].rename(columns={"employee_name":"last_visited_by"})
        agg = agg.merge(last_by, on="client_name", how="left")
    agg["avg_weekly_visits"] = (agg["visits_last_N"] / (window_days/7)).round(2)
    return agg

visit_metrics = compute_visit_metrics(df_visit, window_days=90)

if not sales_orders.empty:
    sales_orders["month"] = sales_orders["order_date"].dt.to_period("M").astype(str)
    revenue_monthly = sales_orders.groupby(["dealer_id","month"]).agg(monthly_revenue=("total_paid_after_tax","sum")).reset_index()
    revenue_total = sales_orders.groupby("dealer_id").agg(total_revenue=("total_paid_after_tax","sum")).reset_index()
    avg_monthly_revenue = revenue_monthly.groupby("dealer_id")["monthly_revenue"].mean().reset_index().rename(columns={"monthly_revenue":"avg_monthly_revenue"})
else:
    revenue_monthly = pd.DataFrame(columns=["dealer_id","month","monthly_revenue"])
    revenue_total = pd.DataFrame(columns=["dealer_id","total_revenue"])
    avg_monthly_revenue = pd.DataFrame(columns=["dealer_id","avg_monthly_revenue"])

run = running_order.copy() if 'running_order' in globals() else pd.DataFrame()
if not run.empty:
    run.columns = [c.strip() for c in run.columns]
    run_ = run.rename(columns={c:c.strip() for c in run.columns})
    id_col = None
    for c in run_.columns:
        if "dealer" in c.lower() and "id" in c.lower():
            id_col = c
    if id_col:
        run_["id_dealer_outlet"] = pd.to_numeric(run_[id_col], errors="coerce")
    if "IsActive" in run_.columns and "End Date" in run_.columns:
        ao = run_[run_["IsActive"].astype(str) == "1"].copy()
        ao["End Date"] = pd.to_datetime(ao["End Date"], errors="coerce")
        ao_group = ao.groupby("id_dealer_outlet").agg(nearest_end_date=("End Date","min")).reset_index()
    else:
        ao_group = pd.DataFrame(columns=["id_dealer_outlet","nearest_end_date"])
    rm = run_.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
    if "id_dealer_outlet" in rm.columns:
        rm["id_dealer_outlet"] = pd.to_numeric(rm["id_dealer_outlet"], errors="coerce")
    rm["joined_dse"] = pd.to_numeric(rm.get("joined_dse"), errors="coerce").fillna(0)
    rm["active_dse"] = pd.to_numeric(rm.get("active_dse"), errors="coerce").fillna(0)
    grouped_run_order = rm.dropna(subset=["id_dealer_outlet"]).groupby(["id_dealer_outlet","dealer_name"]).agg(joined_dse=("joined_dse","count"), active_dse=("active_dse","sum")).reset_index()
    grouped_run_order = grouped_run_order.merge(ao_group, how="left", on="id_dealer_outlet")
else:
    grouped_run_order = pd.DataFrame(columns=["id_dealer_outlet","dealer_name","joined_dse","active_dse","nearest_end_date"])

avail_df = df_dealer.copy()
if "id_dealer_outlet" in avail_df.columns:
    avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce")
if "business_type" in avail_df.columns:
    avail_df = avail_df[avail_df["business_type"].str.lower()=="car"]
if 'location_detail' in globals() and not location_detail.empty:
    ld = location_detail.copy()
    ld.columns = [c.strip() for c in ld.columns]
    mapcols = {}
    for c in ld.columns:
        if c.lower() == "city":
            mapcols[c] = "city"
        if c.lower() == "cluster":
            mapcols[c] = "cluster"
    ld = ld.rename(columns=mapcols)
    if "city" in ld.columns and "cluster" in ld.columns:
        avail_df = avail_df.merge(ld[["city","cluster"]], on="city", how="left")
else:
    avail_df["cluster"] = "General"

avail_df_merge = avail_df.merge(grouped_run_order.drop(columns=["dealer_name"], errors="ignore"), on="id_dealer_outlet", how="left")
vm = visit_metrics.rename(columns={"client_name":"client_name"})
avail_df_merge = avail_df_merge.merge(vm, left_on="client_name", right_on="client_name", how="left")
amr = avg_monthly_revenue.rename(columns={"dealer_id":"id_dealer_outlet"})
avail_df_merge = avail_df_merge.merge(amr, on="id_dealer_outlet", how="left")
rt = revenue_total.rename(columns={"dealer_id":"id_dealer_outlet"})
avail_df_merge = avail_df_merge.merge(rt, on="id_dealer_outlet", how="left")
avail_df_merge["joined_dse"] = avail_df_merge.get("joined_dse", 0).fillna(0).astype(int)
avail_df_merge["active_dse"] = avail_df_merge.get("active_dse", 0).fillna(0).astype(int)
avail_df_merge["total_dse"] = avail_df_merge.get("total_dse", 0).fillna(0).astype(int)
avail_df_merge["visits_last_N"] = avail_df_merge.get("visits_last_N", 0).fillna(0).astype(int)
avail_df_merge["avg_weekly_visits"] = avail_df_merge.get("avg_weekly_visits", 0).fillna(0).astype(float)
avail_df_merge["nearest_end_date"] = avail_df_merge.get("nearest_end_date")
avail_df_merge["avg_monthly_revenue"] = avail_df_merge.get("avg_monthly_revenue").fillna(0) if "avg_monthly_revenue" in avail_df_merge.columns else 0
avail_df_merge["total_revenue"] = avail_df_merge.get("total_revenue").fillna(0) if "total_revenue" in avail_df_merge.columns else 0

def compute_tag(row):
    if row.get("active_dse", 0) > 0:
        return "Active"
    if (row.get("joined_dse",0) == 0) and (row.get("total_dse",0) == 0) and (row.get("visits_last_N",0) == 0):
        return "Not Penetrated"
    if row.get("visits_last_N",0) == 0:
        return "Not Active"
    return "Active"

avail_df_merge["tag"] = avail_df_merge.apply(compute_tag, axis=1)

if 'cluster_left' in globals() and not cluster_left.empty:
    cl = cluster_left.copy()
    cl.columns = [c.strip() for c in cl.columns]
    mapcols = {}
    for c in cl.columns:
        if c.lower() == "cluster":
            mapcols[c] = "cluster"
        if "brand" in c.lower():
            mapcols[c] = "brand"
        if "daily_gen" in c.lower() or "daily gen" in c.lower():
            mapcols[c] = "daily_gen"
        if "daily_need" in c.lower() or "daily need" in c.lower():
            mapcols[c] = "daily_need"
        if "delta" in c.lower():
            mapcols[c] = "delta"
        if "tag" in c.lower():
            mapcols[c] = "availability"
    cl = cl.rename(columns=mapcols)
    if "cluster" in cl.columns and "brand" in cl.columns:
        cl["brand"] = cl["brand"].replace({"CHERY":"Chery","Kia":"KIA"})
        avail_df_merge = avail_df_merge.merge(cl[["cluster","brand","daily_gen","daily_need","delta","availability"]], on=["cluster","brand"], how="left")
else:
    avail_df_merge["availability"] = avail_df_merge.get("availability", "Potential")
