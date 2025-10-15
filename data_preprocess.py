import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from difflib import get_close_matches
from data_load import df_dealer, df_visits_raw, sales_orders, running_order, location_detail, cluster_left

jabodetabek_list = [
    "Bekasi","Bogor","Depok","Jakarta Barat","Jakarta Pusat","Jakarta Selatan","Jakarta Timur","Jakarta Utara",
    "Tangerang","Tangerang Selatan","Cibitung","Tambun","Cikarang","Karawaci","Alam Sutera","Cileungsi","Sentul",
    "Cibubur","Bintaro"
]

def clean_dealers(df):
    df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
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
        if "city" in cl or "kota" in cl:
            colmap[c] = "city"
        if "latitude" in cl or cl == "lat":
            colmap[c] = "latitude"
        if "longitude" in cl or cl in ["long","lon"]:
            colmap[c] = "longitude"
        if "total_dse" in cl or ("total" in cl and "dse" in cl):
            colmap[c] = "total_dse"
        if "business" in cl and "type" in cl:
            colmap[c] = "business_type"
    df = df.rename(columns=colmap)
    if "id_dealer_outlet" in df.columns:
        df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce").astype("Int64")
    else:
        df["id_dealer_outlet"] = pd.Series([pd.NA]*len(df), dtype="Int64")
    for k in ["latitude","longitude"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k].astype(str).str.replace(",","").str.strip("."), errors="coerce")
        else:
            df[k] = np.nan
    if "total_dse" in df.columns:
        df["total_dse"] = pd.to_numeric(df["total_dse"], errors="coerce").fillna(0).astype(int)
    else:
        df["total_dse"] = 0
    if "client_name" not in df.columns:
        df["client_name"] = df.get("name", pd.Series([""]*len(df))).astype(str).str.strip()
    else:
        df["client_name"] = df["client_name"].astype(str).str.strip()
    if "brand" not in df.columns:
        df["brand"] = ""
    if "business_type" in df.columns:
        df["business_type"] = df["business_type"].astype(str).str.title()
    else:
        df["business_type"] = ""
    if "city" not in df.columns:
        df["city"] = ""
    df["city_norm"] = df["city"].astype(str).str.strip().str.title()
    return df

def _parse_combined_ll(s):
    try:
        s = str(s).strip().replace("(","").replace(")","")
        parts = [p.strip() for p in s.split(",") if p.strip()!=""]
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except:
        pass
    return np.nan, np.nan

def clean_visits(df):
    df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    div_col = None
    for c in df.columns:
        cl = c.lower()
        if "tanggal" in cl or "date" in cl or "visit" in cl:
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
            cl = c.lower()
            if ("latlong" in cl) or ("latitude" in cl and "longitude" in cl):
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
    df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
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
        df["dealer_id"] = pd.to_numeric(df["dealer_id"], errors="coerce").astype("Int64")
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

df_dealer = clean_dealers(df_dealer) if "df_dealer" in globals() else pd.DataFrame()
df_visits = clean_visits(df_visits_raw) if "df_visits_raw" in globals() else pd.DataFrame()
sales_orders = clean_orders(sales_orders) if "sales_orders" in globals() else pd.DataFrame()

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    if visits is None or visits.empty:
        return visits
    if dealers is None or dealers.empty:
        visits = visits.copy()
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
            close = get_close_matches(cl, dealer_names, n=1, cutoff=0.80)
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
    visits["client_name_assigned"] = visits["client_name_assigned"].where(
        visits["client_name_assigned"].notna() & visits["client_name_assigned"].astype(str).str.strip().ne(""),
        visits["matched_client"]
    )
    visits["client_name_assigned"] = visits["client_name_assigned"].where(visits["client_name_assigned"].notna(), None)
    return visits

df_visits = assign_visits_to_dealers(df_visits, df_dealer, max_km=1.0)

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
    agg = recent.groupby("client_name").agg(
        visits_last_N=("visit_datetime","count"),
        last_visit_datetime=("visit_datetime","max")
    ).reset_index()
    if "employee_name" in v.columns:
        idx = v.groupby("client_name")["visit_datetime"].idxmax()
        last_by = v.loc[idx, ["client_name","employee_name"]].rename(columns={"employee_name":"last_visited_by"})
        agg = agg.merge(last_by, on="client_name", how="left")
    agg["avg_weekly_visits"] = (agg["visits_last_N"] / (window_days/7)).round(2)
    return agg

visit_metrics = compute_visit_metrics(df_visits, window_days=90)

if not sales_orders.empty:
    sales_orders["month"] = sales_orders["order_date"].dt.to_period("M").astype(str)
    revenue_monthly = sales_orders.groupby(["dealer_id","month"]).agg(monthly_revenue=("total_paid_after_tax","sum")).reset_index()
    revenue_total = sales_orders.groupby("dealer_id").agg(total_revenue=("total_paid_after_tax","sum")).reset_index()
    avg_monthly_revenue = revenue_monthly.groupby("dealer_id")["monthly_revenue"].mean().reset_index().rename(columns={"monthly_revenue":"avg_monthly_revenue"})
else:
    revenue_monthly = pd.DataFrame(columns=["dealer_id","month","monthly_revenue"])
    revenue_total = pd.DataFrame(columns=["dealer_id","total_revenue"])
    avg_monthly_revenue = pd.DataFrame(columns=["dealer_id","avg_monthly_revenue"])

run = running_order.copy() if "running_order" in globals() else pd.DataFrame()
if isinstance(run, pd.DataFrame) and not run.empty:
    run.columns = [c.strip() for c in run.columns]
    id_col = None
    for c in run.columns:
        if "dealer" in c.lower() and "id" in c.lower():
            id_col = c
            break
    rm = run.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
    if id_col and "id_dealer_outlet" not in rm.columns:
        rm["id_dealer_outlet"] = pd.to_numeric(run[id_col], errors="coerce").astype("Int64")
    elif "id_dealer_outlet" in rm.columns:
        rm["id_dealer_outlet"] = pd.to_numeric(rm["id_dealer_outlet"], errors="coerce").astype("Int64")
    rm["joined_dse"] = pd.to_numeric(rm.get("joined_dse"), errors="coerce").fillna(0)
    rm["active_dse"] = pd.to_numeric(rm.get("active_dse"), errors="coerce").fillna(0)
    if "IsActive" in run.columns and "End Date" in run.columns:
        ao = run.copy()
        if id_col and "id_dealer_outlet" not in ao.columns:
            ao["id_dealer_outlet"] = pd.to_numeric(ao[id_col], errors="coerce").astype("Int64")
        elif "id_dealer_outlet" in ao.columns:
            ao["id_dealer_outlet"] = pd.to_numeric(ao["id_dealer_outlet"], errors="coerce").astype("Int64")
        ao = ao[ao["IsActive"].astype(str) == "1"]
        ao["End Date"] = pd.to_datetime(ao["End Date"], errors="coerce")
        ao_group = ao.groupby("id_dealer_outlet").agg(nearest_end_date=("End Date","min")).reset_index()
        ao_group["id_dealer_outlet"] = ao_group["id_dealer_outlet"].astype("Int64")
    else:
        ao_group = pd.DataFrame(columns=["id_dealer_outlet","nearest_end_date"])
        ao_group["id_dealer_outlet"] = ao_group.get("id_dealer_outlet", pd.Series([], dtype="Int64")).astype("Int64")
    if "id_dealer_outlet" in rm.columns:
        grouped_run_order = rm.dropna(subset=["id_dealer_outlet"]).groupby(["id_dealer_outlet","dealer_name"]).agg(
            joined_dse=("joined_dse","count"),
            active_dse=("active_dse","sum")
        ).reset_index()
        grouped_run_order["id_dealer_outlet"] = grouped_run_order["id_dealer_outlet"].astype("Int64")
        if not ao_group.empty:
            grouped_run_order = grouped_run_order.merge(ao_group, how="left", on="id_dealer_outlet")
        else:
            grouped_run_order["nearest_end_date"] = pd.NaT
    else:
        grouped_run_order = pd.DataFrame(columns=["id_dealer_outlet","dealer_name","joined_dse","active_dse","nearest_end_date"])
        grouped_run_order["id_dealer_outlet"] = grouped_run_order.get("id_dealer_outlet", pd.Series([], dtype="Int64")).astype("Int64")
else:
    grouped_run_order = pd.DataFrame(columns=["id_dealer_outlet","dealer_name","joined_dse","active_dse","nearest_end_date"])
    grouped_run_order["id_dealer_outlet"] = grouped_run_order.get("id_dealer_outlet", pd.Series([], dtype="Int64")).astype("Int64")

avail_df = df_dealer.copy()
if "id_dealer_outlet" in avail_df.columns:
    avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce").astype("Int64")
if "business_type" in avail_df.columns:
    avail_df = avail_df[avail_df["business_type"].str.lower().eq("car")]

ld_ok = False
if isinstance(location_detail, pd.DataFrame) and not location_detail.empty:
    ld = location_detail.copy()
    ld.columns = [c.strip() for c in ld.columns]
    cmap = {}
    for c in ld.columns:
        cl = c.lower()
        if cl == "city":
            cmap[c] = "city"
        if cl == "cluster":
            cmap[c] = "cluster"
    ld = ld.rename(columns=cmap)
    if {"city","cluster"}.issubset(ld.columns):
        ld["city_norm"] = ld["city"].astype(str).str.strip().str.title()
        avail_df = avail_df.merge(ld[["city_norm","cluster"]], on="city_norm", how="left")
        ld_ok = True

if not ld_ok:
    avail_df["cluster"] = np.where(avail_df["city_norm"].isin(jabodetabek_list), "Jabodetabek", "Regional")

avail_df_merge = avail_df.merge(grouped_run_order.drop(columns=["dealer_name"], errors="ignore"), on="id_dealer_outlet", how="left")

vm = visit_metrics.rename(columns={"client_name":"client_name"})
avail_df_merge = avail_df_merge.merge(vm, left_on="client_name", right_on="client_name", how="left")

if not sales_orders.empty:
    amr = avg_monthly_revenue.rename(columns={"dealer_id":"id_dealer_outlet"})
    amr["id_dealer_outlet"] = amr["id_dealer_outlet"].astype("Int64")
    avail_df_merge = avail_df_merge.merge(amr, on="id_dealer_outlet", how="left")
    rt = revenue_total.rename(columns={"dealer_id":"id_dealer_outlet"})
    rt["id_dealer_outlet"] = rt["id_dealer_outlet"].astype("Int64")
    avail_df_merge = avail_df_merge.merge(rt, on="id_dealer_outlet", how="left")
else:
    avail_df_merge["avg_monthly_revenue"] = 0.0
    avail_df_merge["total_revenue"] = 0.0

avail_df_merge["joined_dse"] = avail_df_merge.get("joined_dse", 0).fillna(0).astype(int)
avail_df_merge["active_dse"] = avail_df_merge.get("active_dse", 0).fillna(0).astype(int)
avail_df_merge["total_dse"] = avail_df_merge.get("total_dse", 0).fillna(0).astype(int)
avail_df_merge["visits_last_N"] = avail_df_merge.get("visits_last_N", 0).fillna(0).astype(int)
avail_df_merge["avg_weekly_visits"] = avail_df_merge.get("avg_weekly_visits", 0).fillna(0).astype(float)
avail_df_merge["nearest_end_date"] = pd.to_datetime(avail_df_merge.get("nearest_end_date"), errors="coerce")

def compute_tag(row):
    if row.get("active_dse", 0) > 0:
        return "Active"
    if (row.get("joined_dse",0) == 0) and (row.get("total_dse",0) == 0) and (row.get("visits_last_N",0) == 0):
        return "Not Penetrated"
    if row.get("visits_last_N",0) == 0:
        return "Not Active"
    return "Active"

avail_df_merge["tag"] = avail_df_merge.apply(compute_tag, axis=1)

if isinstance(cluster_left, pd.DataFrame) and not cluster_left.empty:
    cl = cluster_left.copy()
    cl.columns = [c.strip() for c in cl.columns]
    cmap = {}
    for c in cl.columns:
        cll = c.lower()
        if cll == "cluster":
            cmap[c] = "cluster"
        if "brand" in cll:
            cmap[c] = "brand"
        if "daily_gen" in cll or "daily gen" in cll:
            cmap[c] = "daily_gen"
        if "daily_need" in cll or "daily need" in cll:
            cmap[c] = "daily_need"
        if "delta" in cll:
            cmap[c] = "delta"
        if "tag" in cll:
            cmap[c] = "availability"
    cl = cl.rename(columns=cmap)
    if {"cluster","brand"}.issubset(cl.columns):
        cl["brand"] = cl["brand"].replace({"CHERY":"Chery","Kia":"KIA"})
        avail_df_merge = avail_df_merge.merge(
            cl[["cluster","brand","daily_gen","daily_need","delta","availability"]],
            on=["cluster","brand"],
            how="left"
        )
else:
    avail_df_merge["availability"] = avail_df_merge.get("availability", "Potential")

try:
    df_visit
except NameError:
    if "df_visits" in globals():
        df_visit = df_visits
    elif "df_visits_raw" in globals():
        df_visit = df_visits_raw
    else:
        df_visit = pd.DataFrame()

if "revenue_monthly" not in globals():
    revenue_monthly = globals().get("revenue_monthly", pd.DataFrame())
