import gspread
import pandas as pd
import streamlit as st

def _client():
    return gspread.service_account_from_dict(st.secrets["google_creds"])

def read_ws_by_id(sheet_id: str, tab: str) -> pd.DataFrame:
    if not sheet_id or not tab:
        raise RuntimeError("Missing sheet id or tab")
    sh = _client().open_by_key(sheet_id)
    ws = sh.worksheet(tab)
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        return pd.DataFrame()
    return pd.DataFrame(vals[1:], columns=vals[0])

@st.cache_data(ttl=300, show_spinner=False)
def get_sheets():
    sids = st.secrets["sheet_ids"]
    need_cluster = read_ws_by_id(sids["need_cluster"], "By Cluster")
    location_detail = read_ws_by_id(sids["package_master"], "City Slug")
    running_order = read_ws_by_id(sids["package_master"], "Database")
    dealers = read_ws_by_id(sids["dealer_book"], "Dealers")
    visits = read_ws_by_id(sids["dealer_book"], "Visits")
    try:
        orders = read_ws_by_id(sids["orders_book"], "Orders")
    except Exception:
        orders = pd.DataFrame()
    return {
        "need_cluster": need_cluster,
        "location_detail": location_detail,
        "running_order": running_order,
        "dealers": dealers,
        "visits": visits,
        "orders": orders,
    }

def clear_cache():
    get_sheets.clear()
