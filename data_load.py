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
        cols = values[0] if values else []
        rows = values[1:] if len(values) > 1 else []
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()

scope = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']
client = gspread.service_account_from_dict(st.secrets["google_creds"])

sid = st.secrets.get("sheet_ids", {})

need_cluster_id = sid.get("need_cluster", "")
location_detail_id = sid.get("location_detail", "")
dealer_penetration_id = sid.get("dealer_penetration", "")
visits_id = sid.get("visits", sid.get("dealer_penetration", ""))
orders_id = sid.get("orders", "")
package_master_id = sid.get("package_master", "")

cluster_left = _ws_to_df(client, need_cluster_id, "By Cluster")
location_detail = _ws_to_df(client, location_detail_id, "Sheet3")
df_dealer = _ws_to_df(client, dealer_penetration_id, "Dealers")
df_visits_raw = _ws_to_df(client, visits_id, "Visits")
sales_orders = _ws_to_df(client, orders_id, "Orders")
running_order = _ws_to_df(client, package_master_id, "Database")

with st.sidebar:
    st.caption("Data status")
    st.write({"cluster_left": len(cluster_left), "location_detail": len(location_detail), "dealers": len(df_dealer), "visits": len(df_visits_raw), "orders": len(sales_orders), "running_order": len(running_order)})
