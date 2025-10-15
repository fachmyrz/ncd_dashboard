import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
import pydeck as pdk
from pydeck.types import String
from data_preprocess import avail_df_merge, df_visit
from datetime import datetime
st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")
st.markdown("<h1 style='font-size:40px;margin-top:0'>Dealer Penetration Dashboard</h1>", unsafe_allow_html=True)
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']
df = avail_df_merge.copy()
df = df[df.get("business_type","Car").str.lower()=="car"] if "business_type" in df.columns else df
sales_names = sorted(list(set(df.get("sales_name", pd.Series([], dtype=str)).dropna().astype(str).tolist() + (df_visit.get("employee_name", pd.Series([], dtype=str)).dropna().astype(str).tolist() if df_visit is not None and not df_visit.empty else []))))
brands_all = sorted(df.get("brand", pd.Series(dtype=str)).dropna().unique().tolist())
cities_all = sorted(df.get("city", pd.Series(dtype=str)).dropna().unique().tolist())
name = st.selectbox("BDE Name", options=["All"] + sales_names, index=0)
area = st.selectbox("Area", options=["Jabodetabek","Regional","All"], index=0)
cols = st.columns(3)
with cols[0]:
    penetrated = st.multiselect("Dealer Activity", options=["Not Active","Not Penetrated","Active"], default=["Not Active","Not Penetrated","Active"])
with cols[1]:
    potential = st.multiselect("Dealer Availability", options=["Potential","Low Generation","Deficit"], default=["Potential","Low Generation","Deficit"])
with cols[2]:
    radius = st.slider("Choose Radius (km)", 0, 50, 15)
if area == "Jabodetabek":
    default_cities = jabodetabek
elif area == "Regional":
    default_cities = [c for c in cities_all if c not in jabodetabek]
else:
    default_cities = cities_all
city_pick = st.multiselect("Choose City", options=["All"] + default_cities, default=["All"])
if "All" in city_pick:
    city_pick = default_cities
brand = st.multiselect("Choose Brand", options=["All"] + brands_all, default=["All"])
if "All" in brand:
    brand = brands_all
button = st.button("Submit")
if button:
    df = avail_df_merge.copy()
    df = df[df.get("business_type","Car").str.lower()=="car"] if "business_type" in df.columns else df
    if name != "All" and "sales_name" in df.columns:
        df = df[df["sales_name"].astype(str) == name]
    if penetrated:
        df = df[df["tag"].isin(penetrated)]
    if potential:
        df = df[df["availability"].isin(potential)]
    if city_pick:
        df = df[df["city"].isin(city_pick)]
    df = df.dropna(subset=["latitude","longitude"])
    if df.empty:
        st.info("No dealers match the filters.")
    else:
        if name != "All":
            center_lat = df["latitude"].mean()
            center_lon = df["longitude"].mean()
            def within_radius_row(r):
                try:
                    lat = float(r["latitude"])
                    lon = float(r["longitude"])
                except:
                    return False
                R = 6371.0
                lat1 = radians(center_lat)
                lon1 = radians(center_lon)
                lat2 = radians(lat)
                lon2 = radians(lon)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.0)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                km = R * c
                return km <= radius
            from math import radians, sin, cos, sqrt, atan2
            df = df[df.apply(within_radius_row, axis=1)]
        df["avg_weekly_visits"] = df.get("avg_weekly_visits", pd.Series(0)).fillna(0)
        df["active_dse"] = df.get("active_dse", pd.Series(0)).fillna(0)
        df["joined_dse"] = df.get("joined_dse", pd.Series(0)).fillna(0)
        df["total_dse"] = df.get("total_dse", pd.Series(0)).fillna(0)
        df["size"] = (df["avg_weekly_visits"] + 1) * 80
        df["map_color"] = df["tag"].map({"Not Penetrated":[220,20,60,200],"Not Active":[255,165,0,200],"Active":[34,139,34,220]}).where(df["tag"].notna(), [200,200,200,160])
        df_zone = df.groupby("city").agg(total_dealers=("id_dealer_outlet","nunique"), active_dealers=("tag", lambda x: (x=="Active").sum()), lat=("latitude","mean"), lon=("longitude","mean")).reset_index()
        df_zone["pct_active"] = (df_zone["active_dealers"] / df_zone["total_dealers"]).fillna(0)
        def color_for_zone(p):
            if p >= 0.75:
                return [34,139,34,60]
            if p >= 0.4:
                return [255,165,0,60]
            return [220,20,60,60]
        df_zone["zone_color"] = df_zone["pct_active"].apply(color_for_zone)
        k1,k2,k3,k4 = st.columns(4)
        with k1:
            st.metric("Dealers", int(df["id_dealer_outlet"].nunique()))
        with k2:
            st.metric("Active Dealers", int(df[df["tag"]=="Active"]["id_dealer_outlet"].nunique()))
        with k3:
            st.metric("Active DSE", int(df["active_dse"].sum()))
        with k4:
            st.metric("Avg Weekly Visits", round(df["avg_weekly_visits"].mean(),2))
        center_lon = float(df["longitude"].mean())
        center_lat = float(df["latitude"].mean())
        zoom = 9 if area == "Jabodetabek" or name == "All" else 11
        tooltip_txt = "Brand: {brand}\nCity: {city}\nActivity: {tag}\nAvg Weekly Visits: {avg_weekly_visits}\nNearest Expiry: {nearest_end_date}"
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=zoom, pitch=40),
            tooltip={"text": tooltip_txt},
            layers=[
                pdk.Layer("ScatterplotLayer", data=df_zone, get_position="[lon,lat]", get_radius="total_dealers*200", get_fill_color="zone_color", pickable=False),
                pdk.Layer("ScatterplotLayer", data=df, get_position="[longitude,latitude]", get_radius="size", get_fill_color="map_color", pickable=True, auto_highlight=True),
                pdk.Layer("TextLayer", data=df_zone, get_position="[lon,lat]", get_text="city", get_size=14, get_color=[0,0,0], get_text_anchor=String("middle"), get_alignment_baseline=String("bottom"))
            ],
        ))
        tab1, tab2 = st.tabs(["Overview","Details"])
        with tab1:
            brand_ct = df.groupby(["brand","tag"]).size().reset_index(name="Count")
            if not brand_ct.empty:
                fig = px.bar(brand_ct, x="brand", y="Count", color="tag")
                st.plotly_chart(fig, use_container_width=True, key="bar_overall")
            pot = df.groupby(["availability","brand"]).size().reset_index(name="Total Dealers")
            if not pot.empty:
                sun = px.sunburst(pot, path=["availability","brand"], values="Total Dealers")
                st.plotly_chart(sun, use_container_width=True, key="sun_overall")
