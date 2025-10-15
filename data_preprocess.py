import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians
from math import sin, cos, sqrt, atan2
from data_load import df_dealer, df_visits_raw, sales_orders, running_order, location_detail, cluster_left

def clean_dealers(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["dealer_id","id_dealer_outlet","id","dealer id","dealer_id"]:
            colmap[c] = "id_dealer_outlet"
        if cl in ["client_name","client name","nama klien","nama_klien"]:
            colmap[c] = "client_name"
        if cl in ["brand","merek"]:
            colmap[c] = "brand"
        if cl in ["city","kota"]:
            colmap[c] = "city"
        if cl in ["name","dealer name","nama dealer","outlet","outlet name"]:
            colmap[c] = "name"
        if cl in ["latitude","lat"]:
            colmap[c] = "latitude"
        if cl in ["longitude","long","lon"]:
            colmap[c] = "longitude"
        if cl in ["total_dse","total dse","total_dse_field","dse_total"]:
            colmap[c] = "total_dse"
        if cl in ["sales_name","bde","bde name","employee name"]:
            colmap[c] = "sales_name"
        if cl in ["business_type","business type","type","kategori","business_type_field"]:
            colmap[c] = "business_type"
    df = df.rename(columns=colmap)
    if "id_dealer_outlet" in df.columns:
        df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce")
    for k in ["latitude","longitude"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k].astype(str).str.replace(",", "").str.strip("."), errors="coerce")
    if "total_dse" not in df.columns:
        df["total_dse"] = 0
    else:
        df["total_dse"] = pd.to_numeric(df["total_dse"], errors="coerce").fillna(0).astype(int)
    if "client_name" in df.columns:
        df["client_name"] = df["client_name"].astype(str).str.strip()
    if "business_type" in df.columns:
        df["business_type"] = df["business_type"].astype(str).str.strip().str.title()
    return df

def _find_combined_latlong_col(df):
    cols = list(df.columns)
    for i,c in enumerate(cols):
        cl = str(c).lower()
        if "lat" in cl and "long" in cl or "latlong" in cl or ("latitude" in cl and "longitude" in cl) or ("&" in cl and "lat" in cl):
            return i, c
    return None, None

def clean_visits(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    div_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ["tanggal datang","tanggal_datang","date","visit date","visited at","tanggal"]:
            colmap[c] = "visit_datetime"
        if cl in ["nama klien","client_name","client name","nama_klien"]:
            colmap[c] = "client_name"
        if cl in ["nama karyawan","employee name","bde","bde name","nama_karyawan"]:
            colmap[c] = "employee_name"
        if cl in ["nomor induk karyawan","nik","employee_id","nomor_induk_karyawan"]:
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
        df["latitude"] = pd.to_numeric(df[lat_col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        df["longitude"] = pd.to_numeric(df[lon_col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    else:
        idx, combined = _find_combined_latlong_col(df)
        if combined is not None:
            s = df.iloc[:, idx].astype(str)
            def parse_ll(x):
                try:
                    t = str(x).strip().replace("(","").replace(")","")
                    parts = [p.strip() for p in t.split(",") if p.strip()!=""]
                    if len(parts) >= 2:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        return lat, lon
                except:
                    return np.nan, np.nan
                return np.nan, np.nan
            parsed = s.apply(parse_ll)
            latitudes = [t[0] if isinstance(t, tuple) else np.nan for t in parsed.tolist()]
            longitudes = [t[1] if isinstance(t, tuple) else np.nan for t in parsed.tolist()]
            df["latitude"] = pd.to_numeric(pd.Series(latitudes, index=df.index), errors="coerce")
            df["longitude"] = pd.to_numeric(pd.Series(longitudes, index=df.index), errors="coerce")
        else:
            df["latitude"] = pd.to_numeric(df.get("latitude", pd.Series([np.nan]*len(df), index=df.index)).astype(str).str.replace(",", ""), errors="coerce")
            df["longitude"] = pd.to_numeric(df.get("longitude", pd.Series([np.nan]*len(df), index=df.index)).astype(str).str.replace(",", ""), errors="coerce")
    return df

def clean_orders(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["dealer_id","id_dealer_outlet","dealer id","id"]:
            colmap[c] = "dealer_id"
        if cl in ["order_date","date","tanggal order","order_date"]:
            colmap[c] = "order_date"
        if cl in ["total_paid_after_tax","amount","total","grand_total"]:
            colmap[c] = "total_paid_after_tax"
    df = df.rename(columns=colmap)
    if "dealer_id" in df.columns:
        df["dealer_id"] = pd.to_numeric(df["dealer_id"], errors="coerce")
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
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

df_dealer = clean_dealers(df_dealer)
df_visits = clean_visits(df_visits_raw)
sales_orders = clean_orders(sales_orders)

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    if visits is None or visits.empty:
        return visits
    if dealers is None or dealers.empty:
        visits["client_name_assigned"] = visits.get("client_name", None)
        return visits
    visits = visits.copy()
    dealers_idx = dealers.dropna(subset=["latitude","longitude"])[["id_dealer_outlet","client_name","latitude","longitude"]].reset_index(drop=True)
    if dealers_idx.empty:
        visits["client_name_assigned"] = visits.get("client_name", None)
        return visits
    dealers_lat = dealers_idx["latitude"].astype(float).to_numpy()
    dealers_lon = dealers_idx["longitude"].astype(float).to_numpy()
    def nearest_client(row):
        try:
            lat = float(row.get("latitude"))
            lon = float(row.get("longitude"))
        except:
            return None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        dists = haversine_km(lat, lon, dealers_lat, dealers_lon)
        if np.all(np.isnan(dists)):
            return None
        idx = int(np.nanargmin(dists))
        min_km = float(dists[idx])
        if np.isfinite(min_km) and min_km <= max_km:
            return dealers_idx.loc[idx, "client_name"]
        return None
    matched = visits.apply(lambda r: nearest_client(r), axis=1)
    visits["matched_client"] = matched
    visits["client_name_assigned"] = visits.get("client_name")
    visits["client_name_assigned"] = visits["client_name_assigned"].where(visits["client_name_assigned"].notna() & visits["client_name_assigned"].astype(str).str.strip().ne(""), visits["matched_client"])
    visits["client_name_assigned"] = visits["client_name_assigned"].where(visits["client_name_assigned"].notna(), None)
    return visits

df_visits = assign_visits_to_dealers(df_visits, df_dealer, max_km=1.0)

def compute_visit_metrics(visits):
    if visits is None or visits.empty:
        return pd.DataFrame(columns=["client_name","visits_last_90","last_visit_datetime","last_visited_by","avg_weekly_visits"])
    v = visits.copy()
    v["client_name"] = v.get("client_name_assigned", v.get("client_name"))
    v = v.dropna(subset=["client_name"])
    v = v[~v["client_name"].astype(str).str.strip().eq("")]
    v["visit_date"] = v["visit_datetime"].dt.date
    today = pd.to_datetime(datetime.utcnow().date())
    window_days = 90
    since = today - pd.Timedelta(days=window_days)
    recent = v[v["visit_datetime"] >= since]
    if recent.empty:
        return pd.DataFrame(columns=["client_name","visits_last_90","last_visit_datetime","last_visited_by","avg_weekly_visits"])
    agg = recent.groupby("client_name").agg(visits_last_90=("visit_datetime","count"), last_visit_datetime=("visit_datetime","max")).reset_index()
    if "employee_name" in v.columns:
        idx = v.groupby("client_name")["visit_datetime"].idxmax()
        last_by = v.loc[idx, ["client_name","employee_name"]].rename(columns={"employee_name":"last_visited_by"})
        agg = agg.merge(last_by, on="client_name", how="left")
    agg["avg_weekly_visits"] = (agg["visits_last_90"] / (window_days/7)).round(2)
    return agg

visit_metrics = compute_visit_metrics(df_visits)

if not sales_orders.empty:
    sales_orders["month"] = sales_orders["order_date"].dt.to_period("M").astype(str)
    revenue_monthly = sales_orders.groupby(["dealer_id","month"]).agg(monthly_revenue=("total_paid_after_tax","sum")).reset_index()
    revenue_total = sales_orders.groupby("dealer_id").agg(total_revenue=("total_paid_after_tax","sum")).reset_index()
    avg_monthly_revenue = revenue_monthly.groupby("dealer_id")["monthly_revenue"].mean().reset_index().rename(columns={"monthly_revenue":"avg_monthly_revenue"})
else:
    revenue_monthly = pd.DataFrame(columns=["dealer_id","month","monthly_revenue"])
    revenue_total = pd.DataFrame(columns=["dealer_id","total_revenue"])
    avg_monthly_revenue = pd.DataFrame(columns=["dealer_id","avg_monthly_revenue"])

run = running_order.copy() if running_order is not None else pd.DataFrame()
if not run.empty:
    run.columns = [c.strip() for c in run.columns]
    ao = run[run.get("IsActive", run.get("IsActive")) == "1"] if "IsActive" in run.columns else run
    ao["End Date"] = pd.to_datetime(ao.get("End Date", ao.get("End_Date", pd.NaT)), errors="coerce")
    ao["Dealer Id"] = pd.to_numeric(ao.get("Dealer Id", ao.get("Dealer_Id", ao.get("id_dealer_outlet"))), errors="coerce")
    ao_group = ao.groupby(["Dealer Id"]).agg(nearest_end_date=("End Date","min")).reset_index().rename(columns={"Dealer Id":"id_dealer_outlet"})
    rm = run.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
    rm["id_dealer_outlet"] = pd.to_numeric(rm.get("id_dealer_outlet"), errors="coerce")
    rm["joined_dse"] = pd.to_numeric(rm.get("joined_dse"), errors="coerce")
    rm["active_dse"] = pd.to_numeric(rm.get("active_dse"), errors="coerce")
    grouped_run_order = rm.dropna(subset=["id_dealer_outlet"]).groupby(["id_dealer_outlet","dealer_name"]).agg(joined_dse=("joined_dse","count"), active_dse=("active_dse","sum")).reset_index()
    grouped_run_order = grouped_run_order.merge(ao_group, how="left", on="id_dealer_outlet")
else:
    grouped_run_order = pd.DataFrame(columns=["id_dealer_outlet","dealer_name","joined_dse","active_dse","nearest_end_date"])

avail_df = df_dealer.copy()
if "id_dealer_outlet" in avail_df.columns:
    avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce")
if "business_type" in avail_df.columns:
    avail_df = avail_df[avail_df["business_type"].str.lower()=="car"]
if not location_detail.empty and "City" in location_detail.columns and "Cluster" in location_detail.columns:
    ld = location_detail.rename(columns={"City":"city","Cluster":"cluster"})
    avail_df = avail_df.merge(ld[["city","cluster"]], on="city", how="left")
else:
    avail_df["cluster"] = "General"

avail_df_merge = avail_df.merge(grouped_run_order.drop(columns=["dealer_name"], errors="ignore"), on="id_dealer_outlet", how="left")
vm = visit_metrics.rename(columns={"client_name":"client_name"})
avail_df_merge = avail_df_merge.merge(vm, on="client_name", how="left")
amr = avg_monthly_revenue.rename(columns={"dealer_id":"id_dealer_outlet"})
avail_df_merge = avail_df_merge.merge(amr, on="id_dealer_outlet", how="left")
rt = revenue_total.rename(columns={"dealer_id":"id_dealer_outlet"})
avail_df_merge = avail_df_merge.merge(rt, on="id_dealer_outlet", how="left")
avail_df_merge["joined_dse"] = avail_df_merge.get("joined_dse", 0).fillna(0)
avail_df_merge["active_dse"] = avail_df_merge.get("active_dse", 0).fillna(0)
avail_df_merge["total_dse"] = avail_df_merge.get("total_dse", 0).fillna(0)
avail_df_merge["visits_last_90"] = avail_df_merge.get("visits_last_90", 0).fillna(0)
avail_df_merge["avg_weekly_visits"] = avail_df_merge.get("avg_weekly_visits", 0).fillna(0)
avail_df_merge["avg_monthly_revenue"] = avail_df_merge.get("avg_monthly_revenue", 0).fillna(0)
avail_df_merge["total_revenue"] = avail_df_merge.get("total_revenue", 0).fillna(0)
avail_df_merge["nearest_end_date"] = avail_df_merge.get("nearest_end_date")

def compute_tag(row):
    if row.get("active_dse", 0) > 0:
        return "Active"
    if (row.get("joined_dse",0) == 0) and (row.get("total_dse",0) == 0) and (row.get("visits_last_90",0) == 0):
        return "Not Penetrated"
    if row.get("visits_last_90",0) == 0:
        return "Not Active"
    return "Active"

avail_df_merge["tag"] = avail_df_merge.apply(compute_tag, axis=1)

if not cluster_left.empty:
    cl = cluster_left.rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"})
    cl["brand"] = cl["brand"].replace({"CHERY":"Chery","Kia":"KIA"})
    avail_df_merge = avail_df_merge.merge(cl[["cluster","brand","daily_gen","daily_need","delta","availability"]], on=["cluster","brand"], how="left")
else:
    avail_df_merge["availability"] = "Potential"
