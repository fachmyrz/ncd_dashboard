import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import requests
import streamlit as st
import json

scope = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']

client = None
try:
    client = gspread.service_account_from_dict(st.secrets["google_creds"])
except Exception as e:
    st.error("gspread client initialization failed: " + str(e))

def safe_load_sheet(client, title, worksheet_name):
    try:
        ws = client.open(title).worksheet(worksheet_name)
        data = ws.get_all_values()
        if len(data) == 0:
            return pd.DataFrame()
        cols = data[0]
        rows = data[1:]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()

if client is not None:
    cluster_left = safe_load_sheet(client, "Gen x Needed Actual Lead Type 71", "By Cluster")
    location_detail = safe_load_sheet(client, "Car Brands Lead Monthly", "Sheet3")
    df_visit = safe_load_sheet(client, "Dealer Penetration Main Data", "Workdata")
    df_dealer = safe_load_sheet(client, "Dealer Penetration Main Data", "Dealer Data")
    sales_data = safe_load_sheet(client, "ID NCD - Sales Dashboard", "NCD Sales Tracker")
    running_order = safe_load_sheet(client, "ID NCD - Package Master", "Database")
else:
    cluster_left = pd.DataFrame()
    location_detail = pd.DataFrame()
    df_visit = pd.DataFrame()
    df_dealer = pd.DataFrame()
    sales_data = pd.DataFrame()
    running_order = pd.DataFrame()

headers_visit = {}
visit_today = pd.DataFrame()
try:
    headers_visit = {
        'accept': 'application/json',
        'Authorization': st.secrets["kerjoo_creds"]["creds"],
    }
    params_visit = {'date_start': '2024-02-22',}
    response_visit = requests.get('https://api.kerjoo.com/tenant11170/api/v1/client-visits', params=params_visit, headers=headers_visit, timeout=10)
    string_visit = response_visit.text
    visit = json.loads(string_visit)
    visit_today = pd.DataFrame(visit.get('data', []))
    if not visit_today.empty and 'personnel' in visit_today.columns:
        visit_today['name'] = visit_today['personnel'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
except Exception:
    visit_today = pd.DataFrame()
