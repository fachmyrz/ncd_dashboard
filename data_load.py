import gspread
import pandas as pd
import streamlit as st

client = gspread.service_account_from_dict(st.secrets["google_creds"])
ids = st.secrets.get("sheet_ids", {})
DEALERS_VISITS_SHEET = ids.get("dealers_visits", "1xGJGwLHqm1bZeT6LWtW0Ah_raHrW-OT9VyAA6ZWhVsk")
ORDERS_SHEET = ids.get("orders", "1QQx6gA8SY8Pj4RZdtkUbeIlyDGDpGMoKdpUSIMGUSzM")
NEED_CLUSTER_SHEET = ids.get("need_cluster")
LOCATION_DETAIL_SHEET = ids.get("location_detail")
PACKAGE_MASTER_SHEET = ids.get("package_master")

def read_ws_by_key(key, tab):
    ss = client.open_by_key(key)
    ws = ss.worksheet(tab)
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()
    return pd.DataFrame(vals[1:], columns=vals[0])

df_dealer = read_ws_by_key(DEALERS_VISITS_SHEET, "Dealers")
df_visits_raw = read_ws_by_key(DEALERS_VISITS_SHEET, "Visits")
sales_orders = read_ws_by_key(ORDERS_SHEET, "Orders")
cluster_left = read_ws_by_key(NEED_CLUSTER_SHEET, "By Cluster") if NEED_CLUSTER_SHEET else pd.DataFrame()
location_detail = read_ws_by_key(LOCATION_DETAIL_SHEET, "Sheet3") if LOCATION_DETAIL_SHEET else pd.DataFrame()
running_order = read_ws_by_key(PACKAGE_MASTER_SHEET, "Database") if PACKAGE_MASTER_SHEET else pd.DataFrame()
