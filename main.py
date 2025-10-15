import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
import pydeck as pdk
from pydeck.types import String
from geopy.distance import geodesic
from data_preprocess import avail_df_merge, revenue_monthly

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="🚗", layout="wide")
st.markdown("<h1 style='font-size:40px;margin-top:0'>Dealer Penetration Dashboard</h1>", unsafe_allow_html=True)

sales_names = sorted(avail_df_merge.get("sales_name", pd.Series(dtype=str)).dropna().unique().tolist()) if "sales_name" in avail_df_merge.columns else []
brands_all = sorted(avail_df_merge.get("brand", pd.Series(dtype=str)).dropna().unique().tolist())
cities_all = sorted(avail_df_merge.get("city", pd.Series(dtype=str)).dropna().unique().tolist())
activities = ["Not Active","Not Penetrated","Active"]
potentials = ["Potential","Low Generation","Deficit"]

name = st.selectbox("BDE Name", options=["All"] + sales_names, index=0)
cols = st.columns(3)
with cols[0]:
    penetrated = st.multiselect("Dealer Activity", options=activities, default=activities)
with cols[1]:
    potential = st.multiselect("Dealer Availability", options=potentials, default=potentials)
with cols[2]:
    radius = st.slider("Choose Radius (km)", 0, 50, 15)
city_pick = st.multiselect("Choose City", options=["All"] + cities_all, default=["All"])
if "All" in city_pick:
    city_pick = cities_all
brand = st.multiselect("Choose Brand", options=["All"] + brands_all, default=["All"])
if "All" in brand:
    brand = brands_all

button = st.button("Submit")

if button:
    df = avail_df_merge.copy()
    if name != "All" and "sales_name" in df.columns:
        df = df[df["sales_name"].astype(str)==name]
    if penetrated:
        df = df[df["tag"].isin(penetrated)]
    if potential:
        df = df[df["availability"].isin(potential)]
    if city_pick:
        df = df[df["city"].isin(city_pick)]
    if brand:
        df = df[df["brand"].isin(brand)]
    df = df.dropna(subset=["latitude","longitude"])

    if not df.empty and name != "All":
        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()
        def d_km(row):
            try:
                return geodesic((center_lat, center_lon), (row["latitude"], row["longitude"])).km
            except:
                return np.nan
        df["dist_center"] = df.apply(d_km, axis=1)
        df = df[df["dist_center"].le(radius)]
    else:
        df["dist_center"] = np.nan

    df["engagement_score"] = (df["avg_weekly_visits"].fillna(0) * 0.6) + (np.log1p(df["total_revenue"].fillna(0)) * 0.4)
    if df["engagement_score"].nunique() >= 3:
        df["engagement_bucket"] = pd.qcut(df["engagement_score"].rank(method="first"), q=3, labels=["Low","Medium","High"])
    else:
        df["engagement_bucket"] = "Low"
    color_map = {"Low":[200,200,200,180],"Medium":[255,165,0,200],"High":[34,139,34,220]}
    df["color"] = df["engagement_bucket"].map(color_map).fillna([200,200,200,180])

    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.metric("Dealers", int(len(df)))
    with k2:
        st.metric("Avg Weekly Visits", round(df["avg_weekly_visits"].mean(),2) if not df.empty else 0)
    with k3:
        st.metric("Joined DSE + Field DSE", int((df.get("joined_dse",0).fillna(0) + df.get("total_dse",0).fillna(0)).sum()) if not df.empty else 0)
    with k4:
        st.metric("Total Revenue", int(df.get("total_revenue",0).fillna(0).sum()) if not df.empty else 0)

    if df.empty:
        st.info("No dealers match the filters.")
    else:
        center_lon = float(df["longitude"].mean())
        center_lat = float(df["latitude"].mean())
        tooltip_txt = "Dealer: {client_name}\nBrand: {brand}\nCity: {city}\nActivity: {tag}\nAvg Weekly Visits: {avg_weekly_visits}\nRevenue: {total_revenue}"
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
            tooltip={"text": tooltip_txt},
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position="[longitude,latitude]",
                    get_radius=220,
                    get_color="color",
                    id="dealer_points",
                    pickable=True,
                    auto_highlight=True
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
            show_cols = ["brand","client_name","city","tag","joined_dse","total_dse","active_dse","avg_weekly_visits","last_visit_datetime","last_visited_by","total_revenue","availability"]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        if not revenue_monthly.empty and "id_dealer_outlet" in df.columns:
            sub = revenue_monthly.merge(df[["id_dealer_outlet"]].drop_duplicates(), left_on="dealer_id", right_on="id_dealer_outlet", how="inner")
            if not sub.empty:
                trend = sub.groupby("month")["monthly_revenue"].sum().reset_index()
                line = px.line(trend, x="month", y="monthly_revenue", markers=True)
                st.plotly_chart(line, use_container_width=True, key="rev_trend")
