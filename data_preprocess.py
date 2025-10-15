import pandas as pd
import numpy as np
from datetime import datetime
from difflib import get_close_matches
from data_load import df_dealer, df_visits_raw, sales_orders, running_order, location_detail, cluster_left

jabodetabek_list = ["Bekasi","Bogor","Depok","Jakarta Barat","Jakarta Pusat","Jakarta Selatan","Jakarta Timur","Jakarta Utara","Tangerang","Tangerang Selatan","Cibitung","Tambun","Cikarang","Karawaci","Alam Sutera","Cileungsi","Sentul","Cibubur","Bintaro"]

def clean_dealers(df):
    df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["dealer id","dealer_id","id_dealer_outlet","id"]:
            colmap[c] = "id_dealer_outlet"
        if ("client" in cl and "name" in cl) or ("nama klien" in cl):
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
    if "id_dealer_outlet" not in df.columns:
        df["id_dealer_outlet"] = pd.Series([pd.NA]*len(df), dtype="Int64")
    df["id_dealer_outlet"] = pd.to_numeric(df["id_dealer_outlet"], errors="coerce").astype("Int64")
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

def clean_visits(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    cand_date_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in ["visit_datetime","tanggal datang","tanggal_datang","visit date","date","date_time_start","date time start","waktu datang","tanggal","order_date","created_at","created at"]:
            cand_date_cols.append(c)
    if not cand_date_cols:
        for c in df.columns:
            if "date" in c.lower():
                cand_date_cols.append(c)
    order_dt = pd.Series([pd.NaT]*len(df))
    for c in cand_date_cols:
        s = pd.to_datetime(df[c], errors="coerce")
        order_dt = order_dt.fillna(s)
    df["visit_datetime"] = order_dt
    def first_col(dfx, names):
        for n in names:
            if n in dfx.columns:
                return n
        return None
    client_col = first_col(df, ["client_name","Nama Klien","nama klien","nama_klien","Client Name"])
    df["client_name"] = df[client_col].astype(str).str.strip() if client_col else ""
    emp_col = first_col(df, ["employee_name","Nama Karyawan","nama karyawan","BDE","bde","Employee Name"])
    df["employee_name"] = df[emp_col].astype(str).str.strip() if emp_col else ""
    nik_col = first_col(df, ["employee_id","Nomor Induk Karyawan","nomor induk karyawan","NIK","nik"])
    df["employee_id"] = df[nik_col].astype(str) if nik_col else ""
    div_col = first_col(df, ["Divisi","divisi","Division","division"])
    if div_col is not None:
        df = df[~df[div_col].astype(str).str.lower().eq("trainer")]
    df = df[~df["employee_id"].astype(str).str.contains("deleted-", na=False)]
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
        comb = None
        for c in df.columns:
            cl = c.lower()
            if ("latlong" in cl) or ("latitude" in cl and "longitude" in cl):
                comb = c
                break
        if comb is not None:
            def _parse_ll(s):
                try:
                    s = str(s).strip().replace("(","").replace(")","")
                    parts = [p.strip() for p in s.split(",") if p.strip()!=""]
                    if len(parts) >= 2:
                        return float(parts[0]), float(parts[1])
                except:
                    pass
                return np.nan, np.nan
            parsed_ll = df[comb].astype(str).apply(_parse_ll)
            df["latitude"] = pd.to_numeric(pd.Series([t[0] for t in parsed_ll], index=df.index), errors="coerce")
            df["longitude"] = pd.to_numeric(pd.Series([t[1] for t in parsed_ll], index=df.index), errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude", pd.Series([np.nan]*len(df))), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", pd.Series([np.nan]*len(df))), errors="coerce")
    df.loc[~df["latitude"].between(-90, 90), "latitude"] = np.nan
    df.loc[~df["longitude"].between(-180, 180), "longitude"] = np.nan
    keep = ["visit_datetime","client_name","employee_name","employee_id","latitude","longitude"]
    keep = [k for k in keep if k in df.columns]
    return df[keep].reset_index(drop=True)

def clean_orders(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["dealer_id","order_date","total_paid_after_tax"])
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    dealer_col = None
    amount_col = None
    date_candidates = []
    for c in df.columns:
        cl = c.lower()
        if dealer_col is None and ("dealer" in cl and "id" in cl):
            dealer_col = c
        if amount_col is None and ("total_paid_after_tax" in cl or cl == "amount" or "total" in cl):
            amount_col = c
        if cl in ["order_date","order date","date","tanggal","tanggal order","tanggal_order","created_at","created at"]:
            date_candidates.append(c)
    if not date_candidates:
        for c in df.columns:
            cl = c.lower()
            if "date" in cl:
                date_candidates.append(c)
    order_dt = pd.Series([pd.NaT]*len(df))
    for c in date_candidates:
        s = pd.to_datetime(df[c], errors="coerce")
        order_dt = order_dt.fillna(s)
    dealer_id = pd.to_numeric(df[dealer_col], errors="coerce").astype("Int64") if dealer_col else pd.Series([pd.NA]*len(df), dtype="Int64")
    amount = pd.to_numeric(df[amount_col], errors="coerce") if amount_col else pd.Series([0.0]*len(df), dtype="float")
    out = pd.DataFrame({"dealer_id": dealer_id, "order_date": order_dt, "total_paid_after_tax": amount})
    return out

def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(pd.to_numeric(lat2_arr, errors="coerce"))
    lon2_r = np.radians(pd.to_numeric(lon2_arr, errors="coerce"))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    if visits is None or visits.empty:
        return visits
    if dealers is None or dealers.empty:
        v = visits.copy()
        v["client_name_assigned"] = v.get("client_name", None)
        return v
    v = visits.copy()
    d = dealers.dropna(subset=["latitude","longitude"]).copy()
    d = d[d["latitude"].between(-90, 90) & d["longitude"].between(-180, 180)]
    d = d[["id_dealer_outlet","client_name","latitude","longitude"]].reset_index(drop=True)
    if d.empty:
        v["client_name_assigned"] = v.get("client_name", None)
        return v
    d["client_lower"] = d["client_name"].astype(str).str.lower()
    names = d["client_lower"].tolist()
    def match_row(row):
        cn = str(row.get("client_name","")).strip()
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        if cn:
            cl = cn.lower()
            if cl in d["client_lower"].values:
                return d.loc[d["client_lower"]==cl, "client_name"].iloc[0]
            close = get_close_matches(cl, names, n=1, cutoff=0.80)
            if close:
                return d.loc[d["client_lower"]==close[0], "client_name"].iloc[0]
        if pd.isna(lat) or pd.isna(lon):
            return None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        dist = haversine_km(lat, lon, d["latitude"].values, d["longitude"].values)
        if dist.size == 0 or np.all(np.isnan(dist)):
            return None
        idx = int(np.nanargmin(dist))
        if dist[idx] <= max_km:
            return d.loc[idx, "client_name"]
        return None
    v["matched_client"] = v.apply(match_row, axis=1)
    v["client_name_assigned"] = v.get("client_name")
    v["client_name_assigned"] = v["client_name_assigned"].where(
        v["client_name_assigned"].notna() & v["client_name_assigned"].astype(str).str.strip().ne(""),
        v["matched_client"]
    )
    v["client_name_assigned"] = v["client_name_assigned"].where(v["client_name_assigned"].notna(), None)
    return v

def compute_visit_metrics(visits, window_days=90):
    if visits is None or visits.empty:
        return pd.DataFrame(columns=["client_name","visits_last_N","last_visit_datetime","last_visited_by","avg_weekly_visits","last_visitor_nik"])
    v = visits.copy()
    v["client_name"] = v.get("client_name_assigned", v.get("client_name"))
    v = v.dropna(subset=["client_name"])
    v = v[~v["client_name"].astype(str).str.strip().eq("")]
    v["visit_datetime"] = pd.to_datetime(v["visit_datetime"], errors="coerce")
    today = pd.to_datetime(datetime.utcnow().date())
    since = today - pd.Timedelta(days=window_days)
    recent = v[v["visit_datetime"] >= since]
    if recent.empty:
        return pd.DataFrame(columns=["client_name","visits_last_N","last_visit_datetime","last_visited_by","avg_weekly_visits","last_visitor_nik"])
    agg = recent.groupby("client_name").agg(
        visits_last_N=("visit_datetime","count"),
        last_visit_datetime=("visit_datetime","max")
    ).reset_index()
    if "employee_name" in v.columns:
        idx = v.groupby("client_name")["visit_datetime"].idxmax()
        last_by = v.loc[idx, ["client_name","employee_name","employee_id"]].rename(columns={"employee_name":"last_visited_by","employee_id":"last_visitor_nik"})
        agg = agg.merge(last_by, on="client_name", how="left")
    agg["avg_weekly_visits"] = (agg["visits_last_N"] / (window_days/7)).round(2)
    return agg

df_dealer = clean_dealers(df_dealer)
df_visits = clean_visits(df_visits_raw)
sales_orders = clean_orders(sales_orders)
df_visits = assign_visits_to_dealers(df_visits, df_dealer, max_km=1.0)
visit_metrics = compute_visit_metrics(df_visits, window_days=90)

run = running_order.copy() if isinstance(running_order, pd.DataFrame) else pd.DataFrame()
if not run.empty:
    run.columns = [c.strip() for c in run.columns]
    rm = run.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
    if "id_dealer_outlet" not in rm.columns:
        id_col = None
        for c in run.columns:
            if "dealer" in c.lower() and "id" in c.lower():
                id_col = c
                break
        rm["id_dealer_outlet"] = pd.to_numeric(run[id_col], errors="coerce").astype("Int64") if id_col else pd.Series([pd.NA]*len(run), dtype="Int64")
    else:
        rm["id_dealer_outlet"] = pd.to_numeric(rm["id_dealer_outlet"], errors="coerce").astype("Int64")
    rm["joined_dse"] = pd.to_numeric(rm.get("joined_dse"), errors="coerce").fillna(0)
    rm["active_dse"] = pd.to_numeric(rm.get("active_dse"), errors="coerce").fillna(0)
    if "IsActive" in run.columns and "End Date" in run.columns:
        ao = run.copy()
        if "id_dealer_outlet" not in ao.columns:
            id_col = None
            for c in run.columns:
                if "dealer" in c.lower() and "id" in c.lower():
                    id_col = c
                    break
            ao["id_dealer_outlet"] = pd.to_numeric(run[id_col], errors="coerce").astype("Int64") if id_col else pd.Series([pd.NA]*len(run), dtype="Int64")
        else:
            ao["id_dealer_outlet"] = pd.to_numeric(ao["id_dealer_outlet"], errors="coerce").astype("Int64")
        ao = ao[ao["IsActive"].astype(str) == "1"]
        ao["End Date"] = pd.to_datetime(ao["End Date"], errors="coerce")
        ao_group = ao.groupby("id_dealer_outlet").agg(nearest_end_date=("End Date","min")).reset_index()
        ao_group["id_dealer_outlet"] = ao_group["id_dealer_outlet"].astype("Int64")
    else:
        ao_group = pd.DataFrame(columns=["id_dealer_outlet","nearest_end_date"])
        ao_group["id_dealer_outlet"] = ao_group.get("id_dealer_outlet", pd.Series([], dtype="Int64")).astype("Int64")
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

avail_df = df_dealer.copy()
avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce").astype("Int64")
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
avail_df_merge = avail_df_merge.merge(vm, on="client_name", how="left")

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
        avail_df_merge = avail_df_merge.merge(cl[["cluster","brand","daily_gen","daily_need","delta","availability"]], on=["cluster","brand"], how="left")

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

df_visit = df_visits.copy()
