import gspread
import pandas as pd
import requests
import streamlit as st

client = gspread.service_account_from_dict(st.secrets["google_creds"])
ids = st.secrets.get("sheet_ids", {})

def open_ws(key_name: str, worksheet_name: str):
    ss = client.open_by_key(ids[key_name])
    return ss.worksheet(worksheet_name)

need_cluster = open_ws("need_cluster", "By Cluster")
data_need_cluster = need_cluster.get_all_values()
cluster_left = pd.DataFrame(data_need_cluster[1:], columns=data_need_cluster[0])

location_sheet = open_ws("location_detail", "Sheet3")
data_location_detail = location_sheet.get_all_values()
location_detail = pd.DataFrame(data_location_detail[1:], columns=data_location_detail[0])

df_visit_ws = open_ws("dealer_penetration", "Workdata")
data_df_visit = df_visit_ws.get_all_values()
df_visit = pd.DataFrame(data_df_visit[1:], columns=data_df_visit[0])

df_dealer_ws = open_ws("dealer_penetration", "Dealer Data")
data_df_dealer = df_dealer_ws.get_all_values()
df_dealer = pd.DataFrame(data_df_dealer[1:], columns=data_df_dealer[0])

sales_data_ws = open_ws("sales_dashboard", "NCD Sales Tracker")
data_sales_data = sales_data_ws.get_all_values()
sales_data = pd.DataFrame(data_sales_data[1:], columns=data_sales_data[0])

running_order_ws = open_ws("package_master", "Database")
data_running_order = running_order_ws.get_all_values()
running_order = pd.DataFrame(data_running_order[1:], columns=data_running_order[0])

kerjoo = st.secrets.get("kerjoo_creds", {})
email = kerjoo.get("email")
password = kerjoo.get("password")
token = kerjoo.get("creds")
params_visit = {"date_start": "2024-02-22"}

try:
    if token:
        headers_visit = {"accept": "application/json", "Authorization": token}
        r = requests.get("https://api.kerjoo.com/tenant11170/api/v1/client-visits", params=params_visit, headers=headers_visit, timeout=60)
    else:
        r = requests.get("https://api.kerjoo.com/tenant11170/api/v1/client-visits", params=params_visit, auth=(email or "", password or ""), timeout=60)
    r.raise_for_status()
    visit_today = pd.DataFrame(r.json().get("data", []))
except Exception:
    visit_today = pd.DataFrame()

if "personnel" in visit_today.columns:
    visit_today["name"] = visit_today["personnel"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
