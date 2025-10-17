import gspread
import pandas as pd
import streamlit as st

def _client():
    creds = st.secrets.get("google_creds")
    if not creds:
        raise RuntimeError("google_creds not found in st.secrets")
    return gspread.service_account_from_dict(creds)

def read_ws_by_id(sheet_id, tab):
    try:
        sh = _client().open_by_key(sheet_id)
        ws = sh.worksheet(tab)
        data = ws.get_all_values()
        if not data:
            return pd.DataFrame()
        cols = data[0]
        rows = data[1:]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()

sids = st.secrets.get("sheet_ids", {})

need_cluster_id = sids.get("need_cluster")
loc_pkg_id = sids.get("package_master")
dealer_file_id = sids.get("dealer_book") or sids.get("dealer_penetration")
orders_file_id = sids.get("orders_book")

cluster_left = read_ws_by_id(need_cluster_id, "By Cluster")
location_detail = read_ws_by_id(loc_pkg_id, "City Slug")
df_dealer = read_ws_by_id(dealer_file_id, "Dealers")
df_visit = read_ws_by_id(dealer_file_id, "Visits")
running_order = read_ws_by_id(loc_pkg_id, "Database")
sales_orders = read_ws_by_id(orders_file_id, "Orders")
