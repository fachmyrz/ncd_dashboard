import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
import pydeck as pdk
from pydeck.types import String
from data_preprocess import avail_df_merge, revenue_monthly, df_visits
from datetime import datetime
from math import isnan

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")
st.markdown("<h1 style='font-size:40px;margin-top:0'>Dealer Penetration Dashboard</h1>", unsafe_allow_html=True)

jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

sales_names_set = set()
if "sales_name" in avail_df_merge.columns:
    sales_names_set.update([x for x in avail_df_merge["sales_name"].dropna().astype(str).unique()])
if df_visits is not None and not df_visits.empty and "employee_name" in df_visits.columns:
    sales_names_set.update([x for x in df_visits["employee_name"].dropna().astype(str).unique()])
sales_names = sorted(list(sales_names_set))
brands_all = sorted(avail_df_merge.get("brand", pd.Series(dtype=str)).dropna().unique().tolist())
cities_all = sorted(avail_df_merge.get("city", pd.Series(dtype=str)).dropna().unique().tolist())

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
    if "business_type" in df.columns:
        df = df[df["business_type"].str.lower()=="car"]
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
            def within_radius(r):
                try:
                    lat = float(r["latitude"])
                    lon = float(r["longitude"])
                except:
                    return False
                from math import radians, sin, cos, sqrt, atan2
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
            df = df[df.apply(within_radius, axis=1)]
        df["avg_weekly_visits"] = df.get("avg_weekly_visits", pd.Series(0)).fillna(0)
        df["avg_monthly_revenue"] = df.get("avg_monthly_revenue", pd.Series(0)).fillna(0)
        df["active_dse"] = df.get("active_dse", pd.Series(0)).fillna(0)
        df["joined_dse"] = df.get("joined_dse", pd.Series(0)).fillna(0)
        df["total_dse"] = df.get("total_dse", pd.Series(0)).fillna(0)
        df["engagement_score"] = (df["avg_weekly_visits"] * 0.6) + (np.log1p(df["avg_monthly_revenue"]) * 0.4)
        if df["engagement_score"].nunique() >= 3:
            df["engagement_bucket"] = pd.qcut(df["engagement_score"].rank(method="first"), q=3, labels=["Low","Medium","High"])
        else:
            df["engagement_bucket"] = "Low"
        engagement_color_map = {"Low":[200,200,200,160],"Medium":[255,165,0,200],"High":[34,139,34,220]}
        df["map_color"] = df["tag"].map({"Not Penetrated":[220,20,60,200],"Not Active":[255,165,0,200],"Active":[34,139,34,220]}).where(df["tag"].notna(), [200,200,200,160])
        df["size"] = (df["avg_weekly_visits"] + 1) * 80
        df["color"] = df["engagement_bucket"].map(engagement_color_map)
        df["color"] = df["color"].apply(lambda x: x if isinstance(x, list) else [200,200,200,160])
        k1,k2,k3,k4 = st.columns(4)
        with k1:
            st.metric("Dealers", int(len(df)))
        with k2:
            st.metric("Active Dealers", int(len(df[df["tag"]=="Active"])))
        with k3:
            st.metric("Active DSE", int(df["active_dse"].sum()))
        with k4:
            st.metric("Avg Weekly Visits", round(df["avg_weekly_visits"].mean(),2))
        center_lon = float(df["longitude"].mean())
        center_lat = float(df["latitude"].mean())
        if area == "Jabodetabek" or name == "All":
            zoom = 9
        else:
            zoom = 11
        tooltip_txt = "Dealer: {client_name}\nBrand: {brand}\nCity: {city}\nActivity: {tag}\nAvg Weekly Visits: {avg_weekly_visits}\nNearest Expiry: {nearest_end_date}"
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=zoom, pitch=40),
            tooltip={"text": tooltip_txt},
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position="[longitude,latitude]",
                    get_radius="size",
                    get_fill_color="map_color",
                    pickable=True,
                    auto_highlight=True
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df[df["tag"]=="Active"],
                    get_position="[longitude,latitude]",
                    get_radius="size",
                    get_fill_color=[34,139,34,220],
                    pickable=False
                ),
                pdk.Layer(
                    "TextLayer",
                    data=df,
                    get_position="[longitude,latitude]",
                    get_text="client_name",
                    get_size=12,
                    get_color=[0,0,0],
                    get_text_anchor=String("middle"),
                    get_alignment_baseline=String("bottom")
                )
            ],
        ))
        tab1, tab2 = st.tabs(["Overview", "Details"])
        with tab1:
            left, right = st.columns([2,1])
            brand_ct = df.groupby(["brand","tag"]).size().reset_index(name="Count")
            if not brand_ct.empty:
                fig = px.bar(brand_ct, x="brand", y="Count", color="tag")
                st.plotly_chart(fig, use_container_width=True, key="bar_overall")
            pot = df.groupby(["availability","brand"]).size().reset_index(name="Total Dealers")
            if not pot.empty:
                sun = px.sunburst(pot, path=["availability","brand"], values="Total Dealers")
                st.plotly_chart(sun, use_container_width=True, key="sun_overall")
        with tab2:
            table = df.copy()
            if "client_name" in table.columns:
                table = table.rename(columns={"client_name":"dealer_name"})
            group_cols = ["id_dealer_outlet"]
            agg_map = {}
            if "dealer_name" in table.columns:
                agg_map["dealer_name"] = lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna())>0 else ""
            if "brand" in table.columns:
                agg_map["brand"] = lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna())>0 else ""
            if "city" in table.columns:
                agg_map["city"] = lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna())>0 else ""
            agg_map["tag"] = lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna())>0 else ""
            agg_map["joined_dse"] = "sum"
            agg_map["total_dse"] = "sum"
            agg_map["active_dse"] = "sum"
            if "avg_weekly_visits" in table.columns:
                agg_map["avg_weekly_visits"] = "mean"
            if "nearest_end_date" in table.columns:
                agg_map["nearest_end_date"] = lambda x: x.dropna().astype(str).min() if len(x.dropna())>0 else ""
            if "last_visit_datetime" in table.columns:
                agg_map["last_visit_datetime"] = "max"
            if "last_visited_by" in table.columns:
                agg_map["last_visited_by"] = lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna())>0 else ""
            if "avg_monthly_revenue" in table.columns:
                agg_map["avg_monthly_revenue"] = "mean"
            table_unique = table.groupby(group_cols).agg(agg_map).reset_index()
            cols_display = ["dealer_name","brand","city","tag","joined_dse","total_dse","active_dse","avg_weekly_visits","last_visit_datetime","last_visited_by","avg_monthly_revenue","nearest_end_date","availability"]
            cols_display = [c for c in cols_display if c in table_unique.columns]
            st.dataframe(table_unique[cols_display].sort_values(by="avg_weekly_visits", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
