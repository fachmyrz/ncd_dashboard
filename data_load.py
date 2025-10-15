import gspread
import pandas as pd
import streamlit as st

def _ws_to_df(client, sheet_id, tab):
    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(tab)
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        cols = values[0]
        rows = values[1:] if len(values) > 1 else []
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()

client = gspread.service_account_from_dict(st.secrets["google_creds"])
sid = st.secrets.get("sheet_ids", {})

need_cluster_id = sid.get("need_cluster", "")
location_detail_id = sid.get("location_detail", "")
dealer_penetration_id = sid.get("dealer_penetration", "")
visits_id = sid.get("visits", dealer_penetration_id)
orders_id = sid.get("orders", "")
package_master_id = sid.get("package_master", "")

cluster_left = _ws_to_df(client, need_cluster_id, "By Cluster")
location_detail = _ws_to_df(client, location_detail_id, "Sheet3")
df_dealer = _ws_to_df(client, dealer_penetration_id, "Dealers")
df_visits_raw = _ws_to_df(client, visits_id, "Visits")
sales_orders = _ws_to_df(client, orders_id, "Orders")
running_order = _ws_to_df(client, package_master_id, "Database")
