import gspread
import pandas as pd
import requests
import streamlit as st
import json

client = gspread.service_account_from_dict(st.secrets["google_creds"])
sid = st.secrets["sheet_ids"]

need_cluster = client.open_by_key(sid["need_cluster"]).worksheet("By Cluster")
location_detail = client.open_by_key(sid["location_detail"]).worksheet("Sheet3")
dealer_book = client.open_by_key(sid["dealer_penetration"])
df_visit_ws = dealer_book.worksheet("Workdata")
df_dealer_ws = dealer_book.worksheet("Dealer Data")
sales_data = client.open_by_key(sid["sales_dashboard"]).worksheet("NCD Sales Tracker")
running_order = client.open_by_key(sid["package_master"]).worksheet("Database")

cluster_left = pd.DataFrame(need_cluster.get_all_values()[1:], columns=need_cluster.get_all_values()[0])
location_detail = pd.DataFrame(location_detail.get_all_values()[1:], columns=location_detail.get_all_values()[0])
df_visit = pd.DataFrame(df_visit_ws.get_all_values()[1:], columns=df_visit_ws.get_all_values()[0])
df_dealer = pd.DataFrame(df_dealer_ws.get_all_values()[1:], columns=df_dealer_ws.get_all_values()[0])
sales_data = pd.DataFrame(sales_data.get_all_values()[1:], columns=sales_data.get_all_values()[0])
running_order = pd.DataFrame(running_order.get_all_values()[1:], columns=running_order.get_all_values()[0])

headers_visit = {"accept": "application/json", "Authorization": st.secrets["kerjoo_creds"].get("creds", "")}
params_visit = {"date_start": "2024-02-22"}
response_visit = requests.get("https://api.kerjoo.com/tenant11170/api/v1/client-visits", params=params_visit, headers=headers_visit)
try:
    visit_today = pd.DataFrame(json.loads(response_visit.text).get("data", []))
    if "personnel" in visit_today.columns:
        visit_today["name"] = visit_today["personnel"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
    else:
        visit_today["name"] = None
except Exception:
    visit_today = pd.DataFrame(columns=["name"])
