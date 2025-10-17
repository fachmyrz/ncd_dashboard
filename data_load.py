# data_load.py
import gspread
import pandas as pd
import streamlit as st
from typing import Dict

@st.cache_data(ttl=300)  # cache for 5 minutes to avoid repeated slow HTTP calls
def _client():
    creds = st.secrets.get("google_creds")
    if not creds:
        raise RuntimeError("Missing google_creds in Streamlit secrets.")
    return gspread.service_account_from_dict(creds)

def read_ws_by_id(sheet_id: str, tab: str) -> pd.DataFrame:
    """Return DataFrame for exact tab name. Empty DataFrame if anything fails."""
    if not sheet_id or not tab:
        return pd.DataFrame()
    try:
        client = _client()
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(tab)
        vals = ws.get_all_values()
        if not vals:
            return pd.DataFrame()
        cols = vals[0]
        rows = vals[1:]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_all_sheets() -> Dict[str, pd.DataFrame]:
    sids = st.secrets.get("sheet_ids", {})
    return {
        "cluster_left": read_ws_by_id(sids.get("need_cluster",""), "By Cluster"),
        "location_detail": read_ws_by_id(sids.get("package_master",""), "City Slug"),
        "df_dealer": read_ws_by_id(sids.get("dealer_book",""), "Dealers"),
        "df_visit": read_ws_by_id(sids.get("dealer_book",""), "Visits"),
        "running_order": read_ws_by_id(sids.get("package_master",""), "Database"),
        "sales_orders": read_ws_by_id(sids.get("orders_book",""), "Orders")
    }

# convenience: a single cached call
def get_sheets():
    return load_all_sheets()
