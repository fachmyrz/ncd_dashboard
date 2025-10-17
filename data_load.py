# data_load.py
import gspread
import pandas as pd
import streamlit as st

def _client():
    creds = st.secrets.get("google_creds")
    if not creds:
        raise RuntimeError("Missing google_creds in Streamlit secrets.")
    return gspread.service_account_from_dict(creds)

def read_ws_by_id(sheet_id: str, tab: str) -> pd.DataFrame:
    """
    Read an exact tab name from a Google Sheet by key.
    Returns empty DataFrame if sheet/tab not found or empty.
    """
    if not sheet_id or not tab:
        return pd.DataFrame()
    gc = _client()
    try:
        sh = gc.open_by_key(sheet_id)
    except Exception:
        return pd.DataFrame()
    try:
        ws = sh.worksheet(tab)
    except Exception:
        return pd.DataFrame()
    data = ws.get_all_values()
    if not data:
        return pd.DataFrame()
    cols = data[0]
    rows = data[1:]
    return pd.DataFrame(rows, columns=cols)

# exact names expected in secrets["sheet_ids"]
sids = st.secrets.get("sheet_ids", {})

# exact tab names you insisted on
cluster_left = read_ws_by_id(sids.get("need_cluster",""), "By Cluster")
location_detail = read_ws_by_id(sids.get("package_master",""), "City Slug")
df_dealer = read_ws_by_id(sids.get("dealer_book",""), "Dealers")
df_visit = read_ws_by_id(sids.get("dealer_book",""), "Visits")
running_order = read_ws_by_id(sids.get("package_master",""), "Database")
sales_orders = read_ws_by_id(sids.get("orders_book",""), "Orders")
