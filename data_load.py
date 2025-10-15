import gspread
import pandas as pd
import requests
import streamlit as st
import json

client = gspread.service_account_from_dict(st.secrets["google_creds"])
sid = st.secrets["sheet_ids"]

need_cluster_ws = client.open_by_key(sid["need_cluster"]).worksheet("By Cluster")
location_detail_ws = client.open_by_key(sid["location_detail"]).worksheet("Sheet3")
dealer_book = client.open_by_key(sid["dealer_book"])
dealers_ws = dealer_book.worksheet("Dealers")
visits_ws = dealer_book.worksheet("Visits")
orders_ws = client.open_by_key(sid["orders_book"]).worksheet("Orders")
running_order_ws = client.open_by_key(sid["package_master"]).worksheet("Database")

nc_vals = need_cluster_ws.get_all_values()
ld_vals = location_detail_ws.get_all_values()
dealers_vals = dealers_ws.get_all_values()
visits_vals = visits_ws.get_all_values()
orders_vals = orders_ws.get_all_values()
running_vals = running_order_ws.get_all_values()

cluster_left = pd.DataFrame(nc_vals[1:], columns=nc_vals[0]) if nc_vals else pd.DataFrame()
location_detail = pd.DataFrame(ld_vals[1:], columns=ld_vals[0]) if ld_vals else pd.DataFrame()
df_dealer = pd.DataFrame(dealers_vals[1:], columns=dealers_vals[0]) if dealers_vals else pd.DataFrame()
df_visit = pd.DataFrame(visits_vals[1:], columns=visits_vals[0]) if visits_vals else pd.DataFrame()
sales_orders = pd.DataFrame(orders_vals[1:], columns=orders_vals[0]) if orders_vals else pd.DataFrame()
running_order = pd.DataFrame(running_vals[1:], columns=running_vals[0]) if running_vals else pd.DataFrame()

headers_visit = {"accept": "application/json", "Authorization": st.secrets["kerjoo_creds"].get("creds", "")}
params_visit = {"date_start": "2024-02-22"}
try:
    r = requests.get("https://api.kerjoo.com/tenant11170/api/v1/client-visits", params=params_visit, headers=headers_visit, timeout=10)
    payload = json.loads(r.text) if r.ok else {}
    visit_today = pd.DataFrame(payload.get("data", []))
    if "personnel" in visit_today.columns:
        visit_today["name"] = visit_today["personnel"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
    else:
        visit_today["name"] = None
except Exception:
    visit_today = pd.DataFrame(columns=["name"])
