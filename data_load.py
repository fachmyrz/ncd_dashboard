import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st

def _client():
    return gspread.service_account_from_dict(st.secrets["google_creds"])

def read_ws_by_id(sheet_id, tab):
    sh = _client().open_by_key(sheet_id)
    ws = sh.worksheet(tab)
    data = ws.get_all_values()
    if not data:
        return pd.DataFrame()
    cols = data[0]
    rows = data[1:]
    return pd.DataFrame(rows, columns=cols)

need_cluster_id = st.secrets["sheet_ids"]["need_cluster"]
loc_pkg_id = st.secrets["sheet_ids"]["package_master"]
dealer_file_id = st.secrets["sheet_ids"]["dealer_penetration"]

cluster_left = read_ws_by_id(need_cluster_id, "By Cluster")
location_detail = read_ws_by_id(loc_pkg_id, "City Slug")
df_dealer = read_ws_by_id(dealer_file_id, "Dealers")
df_visit = read_ws_by_id(dealer_file_id, "Visits")
running_order = read_ws_by_id(loc_pkg_id, "Database")
