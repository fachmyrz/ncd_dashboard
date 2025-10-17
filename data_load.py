import gspread
import pandas as pd
import streamlit as st

def _client():
    return gspread.service_account_from_dict(st.secrets["google_creds"])

def _read(sheet_id, tab):
    try:
        sh = _client().open_by_key(sheet_id)
        ws = sh.worksheet(tab)
        vals = ws.get_all_values()
        if not vals or len(vals) < 2:
            return pd.DataFrame()
        return pd.DataFrame(vals[1:], columns=vals[0])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_sheets():
    sids = st.secrets["sheet_ids"]
    need_cluster = _read(sids["need_cluster"], "By Cluster")
    location_detail = _read(sids["package_master"], "City Slug")
    running_order = _read(sids["package_master"], "Database")
    dealers = _read(sids["dealer_book"], "Dealers")
    visits = _read(sids["dealer_book"], "Visits")
    try:
        orders = _read(sids["orders_book"], "Orders")
    except Exception:
        orders = pd.DataFrame()
    return {"need_cluster": need_cluster, "location_detail": location_detail, "running_order": running_order, "dealers": dealers, "visits": visits, "orders": orders}
