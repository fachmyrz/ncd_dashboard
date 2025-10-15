import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import requests
import streamlit as st
import json

def _get_client():
    creds = st.secrets.get("google_creds", None)
    if creds is None:
        return None
    client = gspread.service_account_from_dict(creds)
    return client

def _sheet_to_df_by_key(client, key, tab):
    try:
        sh = client.open_by_key(key)
        ws = sh.worksheet(tab)
        data = ws.get_all_values()
        if not data:
            return pd.DataFrame()
        cols = data[0]
        rows = data[1:]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()

client = _get_client()

sheet_ids = st.secrets.get("sheet_ids", {})

need_cluster = _sheet_to_df_by_key(client, sheet_ids.get("need_cluster", ""), "By Cluster")
location_detail = _sheet_to_df_by_key(client, sheet_ids.get("location_detail", ""), "Sheet3")
dealer_penetration = _sheet_to_df_by_key(client, sheet_ids.get("dealer_penetration", ""), "Dealers")
sales_dashboard = _sheet_to_df_by_key(client, sheet_ids.get("sales_dashboard", ""), "NCD Sales Tracker")
package_master = _sheet_to_df_by_key(client, sheet_ids.get("package_master", ""), "Database")

df_dealer = dealer_penetration.copy()
sales_data = sales_dashboard.copy()
running_order = package_master.copy()
cluster_left = need_cluster.copy()
location_detail = location_detail.copy()

kerjoo = st.secrets.get("kerjoo_creds", None)
df_visits_raw = pd.DataFrame()
if kerjoo:
    token = kerjoo.get("creds") or kerjoo.get("token")
    if token:
        headers_visit = {
            "accept": "application/json",
            "Authorization": token
        }
        try:
            params_visit = {"date_start": "2024-02-22"}
            response_visit = requests.get("https://api.kerjoo.com/tenant11170/api/v1/client-visits", params=params_visit, headers=headers_visit, timeout=10)
            if response_visit.status_code == 200:
                j = response_visit.json()
                df_visits_raw = pd.DataFrame(j.get("data", []))
                if not df_visits_raw.empty:
                    if "personnel" in df_visits_raw.columns:
                        df_visits_raw["name_personnel"] = df_visits_raw["personnel"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
                    if "client" in df_visits_raw.columns:
                        df_visits_raw["client_name_raw"] = df_visits_raw["client"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
        except Exception:
            df_visits_raw = pd.DataFrame()
if df_visits_raw.empty:
    df_visits_raw = pd.DataFrame()

orders_sheet_key = st.secrets.get("sheet_ids", {}).get("sales_dashboard", "")
sales_orders = pd.DataFrame()
if client and orders_sheet_key:
    sales_orders = _sheet_to_df_by_key(client, orders_sheet_key, "Orders")
