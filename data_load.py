import json
import requests
import pandas as pd
import gspread
import streamlit as st

# -------- Google Sheets Client --------
@st.cache_data(show_spinner=False)
def get_gspread_client():
    """
    Builds a Google Sheets client from Streamlit secrets.
    """
    # Expecting st.secrets["google_creds"] to be a full service account JSON dict
    client = gspread.service_account_from_dict(st.secrets["google_creds"])
    return client

@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_sheet(spreadsheet_name: str, worksheet_name: str) -> pd.DataFrame:
    """
    Load a single worksheet to a DataFrame.
    """
    client = get_gspread_client()
    ws = client.open(spreadsheet_name).worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    cols = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=cols)

# -------- Domain Loaders (Sheets) --------
@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_cluster_left() -> pd.DataFrame:
    # "Gen x Needed Actual Lead Type 71" / 'By Cluster'
    return load_sheet("Gen x Needed Actual Lead Type 71", "By Cluster")

@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_location_detail() -> pd.DataFrame:
    # "Car Brands Lead Monthly" / 'Sheet3'
    return load_sheet("Car Brands Lead Monthly", "Sheet3")

@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_df_visit() -> pd.DataFrame:
    # "Dealer Penetration Main Data" / 'Workdata'
    return load_sheet("Dealer Penetration Main Data", "Workdata")

@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_df_dealer() -> pd.DataFrame:
    # "Dealer Penetration Main Data" / 'Dealer Data'
    return load_sheet("Dealer Penetration Main Data", "Dealer Data")

@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_sales_data() -> pd.DataFrame:
    # "ID NCD - Sales Dashboard" / 'NCD Sales Tracker'
    return load_sheet("ID NCD - Sales Dashboard", "NCD Sales Tracker")

@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_running_order() -> pd.DataFrame:
    # "ID NCD - Package Master" / 'Database'
    return load_sheet("ID NCD - Package Master", "Database")

# -------- Kerjoo API (optional) --------
@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_visit_today(date_start: str = "2024-02-22") -> pd.DataFrame:
    """
    Pulls visit data from Kerjoo API.
    Needs st.secrets['kerjoo_creds']['creds'] = 'Bearer ...'
    """
    headers_visit = {
        "accept": "application/json",
        "Authorization": st.secrets["kerjoo_creds"]["creds"],
    }
    params_visit = {"date_start": date_start}
    r = requests.get(
        "https://api.kerjoo.com/tenant11170/api/v1/client-visits",
        params=params_visit,
        headers=headers_visit,
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload.get("data", []))
    if "personnel" in df.columns:
        df["name"] = df["personnel"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
    return df

# -------- Convenience: load everything needed for preprocessing --------
@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_all_inputs():
    return {
        "cluster_left": load_cluster_left(),
        "location_detail": load_location_detail(),
        "df_visit": load_df_visit(),
        "df_dealer": load_df_dealer(),
        "sales_data": load_sales_data(),
        "running_order": load_running_order(),
        # Optional:
        "visit_today": load_visit_today(),
    }
