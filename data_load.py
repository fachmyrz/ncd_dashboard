import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import requests
import streamlit as st
import json

scope = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']
client = gspread.service_account_from_dict(st.secrets["google_creds"])

need_cluster = client.open("Gen x Needed Actual Lead Type 71").worksheet('By Cluster')
data_need_cluster = need_cluster.get_all_values()
column_names_need_cluster = data_need_cluster[0]
data_need_cluster = data_need_cluster[1:]
cluster_left = pd.DataFrame(data_need_cluster, columns=column_names_need_cluster)

location_detail = client.open("Car Brands Lead Monthly").worksheet('Sheet3')
data_location_detail = location_detail.get_all_values()
column_names_location_detail = data_location_detail[0]
data_location_detail = data_location_detail[1:]
location_detail = pd.DataFrame(data_location_detail, columns=column_names_location_detail)

df_visit_ws = client.open("Dealer Penetration Main Data").worksheet('Workdata')
data_df_visit = df_visit_ws.get_all_values()
column_names_df_visit = data_df_visit[0]
data_df_visit = data_df_visit[1:]
df_visit = pd.DataFrame(data_df_visit, columns=column_names_df_visit)

df_dealer_ws = client.open("Dealer Penetration Main Data").worksheet('Dealer Data')
data_df_dealer = df_dealer_ws.get_all_values()
column_names_df_dealer = data_df_dealer[0]
data_df_dealer = data_df_dealer[1:]
df_dealer = pd.DataFrame(data_df_dealer, columns=column_names_df_dealer)

sales_data_ws = client.open("ID NCD - Sales Dashboard").worksheet('NCD Sales Tracker')
data_sales_data = sales_data_ws.get_all_values()
column_names_sales_data = data_sales_data[0]
data_sales_data = data_sales_data[1:]
sales_data = pd.DataFrame(data_sales_data, columns=column_names_sales_data)

running_order_ws = client.open("ID NCD - Package Master").worksheet('Database')
data_running_order = running_order_ws.get_all_values()
column_names_running_order = data_running_order[0]
data_running_order = data_running_order[1:]
running_order = pd.DataFrame(data_running_order, columns=column_names_running_order)

kerjoo = st.secrets.get("kerjoo_creds", {})
email = kerjoo.get("email")
password = kerjoo.get("password")
token = kerjoo.get("creds")
params_visit = {'date_start': '2024-02-22'}
try:
    if token:
        headers_visit = {'accept': 'application/json','Authorization': token}
        response_visit = requests.get('https://api.kerjoo.com/tenant11170/api/v1/client-visits', params=params_visit, headers=headers_visit, timeout=60)
    else:
        response_visit = requests.get('https://api.kerjoo.com/tenant11170/api/v1/client-visits', params=params_visit, auth=(email or "", password or ""), timeout=60)
    response_visit.raise_for_status()
    visit = response_visit.json()
    visit_today = pd.DataFrame(visit.get('data', []))
except Exception:
    visit_today = pd.DataFrame()

if 'personnel' in visit_today.columns:
    visit_today['name'] = visit_today['personnel'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
