import gspread
import pandas as pd
import streamlit as st

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

def get_sources():
    need_cluster = _read_by_title("Gen x Needed Actual Lead Type 71", "By Cluster")
    location_detail = _read_by_title("Car Brands Lead Monthly", "Sheet3")
    df_visit = _read_by_title("Dealers Directory", "Visits")
    df_dealer = _read_by_title("Dealers Directory", "Dealers")
    orders = _read_by_title("ID NCD - Order Dashboard", "Orders")
    running_order = _read_by_title("ID NCD - Package Master", "Database")
    return {"need_cluster": need_cluster, "location_detail": location_detail, "df_visit": df_visit, "df_dealer": df_dealer, "orders": orders, "running_order": running_order}
