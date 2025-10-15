import pandas as pd
import numpy as np
import geopy.distance
from datetime import datetime, timedelta
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
    return df

def clean_visits(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
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
        if "latitude" in cl and "longitude" in cl:
            colmap[c] = "latlong"
        if cl in ["latitude & longitude datang","latlong datang","latlong_datang"]:
            colmap[c] = "latlong"
    df = df.rename(columns=colmap)
    if "visit_datetime" in df.columns:
        df["visit_datetime"] = pd.to_datetime(df["visit_datetime"], errors="coerce")
    else:
        df["visit_datetime"] = pd.NaT
    if "client_name" in df.columns:
        df["client_name"] = df["client_name"].astype(str).str.strip()
    if "employee_id" in df.columns:
        df = df[~df["employee_id"].astype(str).str.contains("deleted-", na=False)]
    if "latlong" in df.columns:
        def parse_ll(x):
            s = str(x)
            s = s.replace("(", "").replace(")", "")
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 2:
                try:
                    return float(parts[0]), float(parts[1])
                except:
                    return np.nan, np.nan
            return np.nan, np.nan
        ll = df["latlong"].apply(parse_ll)
        df["latitude"] = ll.apply(lambda t: t[0])
        df["longitude"] = ll.apply(lambda t: t[1])
    return df

def clean_orders(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["dealer_id","id_dealer_outlet","dealer id","id"]:
            colmap[c] = "dealer_id"
        if cl in ["order_date","date","tanggal order"]:
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

df_dealer = clean_dealers(df_dealer)
df_visits = clean_visits(df_visits_raw)
sales_orders = clean_orders(sales_orders)

def compute_visit_metrics(visits):
    v = visits.dropna(subset=["client_name"]).copy()
    v = v[~v["client_name"].astype(str).str.strip().eq("")]
    v["visit_date"] = v["visit_datetime"].dt.date
    today = pd.to_datetime(datetime.utcnow().date())
    window_days = 90
    since = today - pd.Timedelta(days=window_days)
    recent = v[v["visit_datetime"] >= since]
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
else:
    revenue_monthly = pd.DataFrame(columns=["dealer_id","month","monthly_revenue"])
    revenue_total = pd.DataFrame(columns=["dealer_id","total_revenue"])

run = running_order.copy() if running_order is not None else pd.DataFrame()
if not run.empty:
    run.columns = [c.strip() for c in run.columns]
    rm = run.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse"})
    rm["id_dealer_outlet"] = pd.to_numeric(rm.get("id_dealer_outlet"), errors="coerce")
    rm["joined_dse"] = pd.to_numeric(rm.get("joined_dse"), errors="coerce")
    rm["active_dse"] = pd.to_numeric(rm.get("active_dse"), errors="coerce")
    grouped_run_order = rm.dropna(subset=["id_dealer_outlet"]).groupby(["id_dealer_outlet","dealer_name"]).agg(joined_dse=("joined_dse","count"), active_dse=("active_dse","sum")).reset_index()
else:
    grouped_run_order = pd.DataFrame(columns=["id_dealer_outlet","dealer_name","joined_dse","active_dse"])

avail_df = df_dealer.copy()
if "id_dealer_outlet" in avail_df.columns:
    avail_df["id_dealer_outlet"] = pd.to_numeric(avail_df["id_dealer_outlet"], errors="coerce")
if not location_detail.empty and "City" in location_detail.columns and "Cluster" in location_detail.columns:
    ld = location_detail.rename(columns={"City":"city","Cluster":"cluster"})
    avail_df = avail_df.merge(ld[["city","cluster"]], on="city", how="left")
else:
    avail_df["cluster"] = "General"

avail_df_merge = avail_df.merge(grouped_run_order.drop(columns=["dealer_name"], errors="ignore"), on="id_dealer_outlet", how="left")
vm = visit_metrics.rename(columns={"client_name":"client_name"})
avail_df_merge = avail_df_merge.merge(vm, on="client_name", how="left")
rt = revenue_total.rename(columns={"dealer_id":"id_dealer_outlet"})
avail_df_merge = avail_df_merge.merge(rt, on="id_dealer_outlet", how="left")
avail_df_merge["joined_dse"] = avail_df_merge.get("joined_dse", 0).fillna(0)
avail_df_merge["active_dse"] = avail_df_merge.get("active_dse", 0).fillna(0)
avail_df_merge["total_dse"] = avail_df_merge.get("total_dse", 0).fillna(0)
avail_df_merge["visits_last_90"] = avail_df_merge.get("visits_last_90", 0).fillna(0)
avail_df_merge["avg_weekly_visits"] = avail_df_merge.get("avg_weekly_visits", 0).fillna(0)
avail_df_merge["total_revenue"] = avail_df_merge.get("total_revenue", 0).fillna(0)
def compute_tag(row):
    if row["joined_dse"]==0 and row["total_dse"]==0 and row["visits_last_90"]==0:
        return "Not Penetrated"
    if row["visits_last_90"]==0:
        return "Not Active"
    return "Active"
avail_df_merge["tag"] = avail_df_merge.apply(compute_tag, axis=1)
if not cluster_left.empty:
    cl = cluster_left.rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"})
    cl["brand"] = cl["brand"].replace({"CHERY":"Chery","Kia":"KIA"})
    avail_df_merge = avail_df_merge.merge(cl[["cluster","brand","daily_gen","daily_need","delta","availability"]], on=["cluster","brand"], how="left")
else:
    avail_df_merge["availability"] = "Potential"
