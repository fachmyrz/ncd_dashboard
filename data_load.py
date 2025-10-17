# data_load.py
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st

def _client():
    # expects st.secrets["google_creds"] to contain a service account dict
    creds = st.secrets.get("google_creds")
    if not creds:
        raise RuntimeError("Missing google_creds in Streamlit secrets.")
    return gspread.service_account_from_dict(creds)

def read_ws_by_id(sheet_id: str, tab: str) -> pd.DataFrame:
    """
    Read exact worksheet name first; if not found, try case-insensitive match among worksheets.
    Returns empty DataFrame if none found or sheet has no rows.
    """
    if not sheet_id:
        return pd.DataFrame()
    gc = _client()
    try:
        sh = gc.open_by_key(sheet_id)
    except Exception:
        return pd.DataFrame()
    # Try exact first
    try:
        ws = sh.worksheet(tab)
    except Exception:
        # fallback: find a worksheet whose title matches case-insensitively or contains the token
        titles = [w.title for w in sh.worksheets()]
        match = None
        for t in titles:
            if t.lower() == str(tab).lower():
                match = t
                break
        if not match:
            for t in titles:
                if str(tab).lower() in t.lower():
                    match = t
                    break
        if not match:
            return pd.DataFrame()
        ws = sh.worksheet(match)
    data = ws.get_all_values()
    if not data:
        return pd.DataFrame()
    cols = data[0]
    rows = data[1:]
    return pd.DataFrame(rows, columns=cols)

# keys from st.secrets["sheet_ids"]
sids = st.secrets.get("sheet_ids", {})

need_cluster_id = sids.get("need_cluster", "")
loc_pkg_id = sids.get("package_master", "")
dealer_file_id = sids.get("dealer_book", "") or sids.get("dealer_penetration", "")
orders_file_id = sids.get("orders_book", "")

cluster_left = read_ws_by_id(need_cluster_id, "By Cluster")
location_detail = read_ws_by_id(loc_pkg_id, "City Slug")
df_dealer = read_ws_by_id(dealer_file_id, "Dealers")
df_visit = read_ws_by_id(dealer_file_id, "Visits")
running_order = read_ws_by_id(loc_pkg_id, "Database")
sales_orders = read_ws_by_id(orders_file_id, "Orders") if orders_file_id else pd.DataFrame()
