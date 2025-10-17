import gspread
import pandas as pd
import requests
import streamlit as st
import json

def _client():
    return gspread.service_account_from_dict(st.secrets["google_creds"])

def _read_by_title(book_title: str, tab: str) -> pd.DataFrame:
    try:
        sh = _client().open(book_title)
        ws = sh.worksheet(tab)
        vals = ws.get_all_values()
        if not vals or len(vals) < 2:
            return pd.DataFrame()
        return pd.DataFrame(vals[1:], columns=[c.strip() for c in vals[0]])
    except Exception:
        return pd.DataFrame()

cluster_left = _read_by_title("Gen x Needed Actual Lead Type 71", "By Cluster")
location_detail = _read_by_title("Car Brands Lead Monthly", "Sheet3")
df_visit = _read_by_title("Dealers Directory", "Visits")
df_dealer = _read_by_title("Dealers Directory", "Dealers")
sales_data = _read_by_title("ID NCD - Order Dashboard", "Orders")
running_order = _read_by_title("ID NCD - Package Master", "Database")

try:
    headers_visit = {"accept": "application/json", "Authorization": st.secrets["kerjoo_creds"]["creds"]}
    params_visit = {"date_start": "2024-02-22"}
    response_visit = requests.get("https://api.kerjoo.com/tenant11170/api/v1/client-visits", params=params_visit, headers=headers_visit, timeout=30)
    obj = json.loads(response_visit.text)
    visit_today = pd.DataFrame(obj.get("data", []))
    if "personnel" in visit_today.columns:
        visit_today["name"] = visit_today["personnel"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
except Exception:
    visit_today = pd.DataFrame()
