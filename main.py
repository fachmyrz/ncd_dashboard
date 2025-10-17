# main.py
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pydeck as pdk
from pydeck.types import String
import plotly.express as px

from data_preprocess import compute_all

# page config & favicon
try:
    icon = Image.open("assets/favicon.png")
    st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=icon, layout="wide")
except:
    st.set_page_config(page_title="Dealer Penetration Dashboard", layout="wide")

st.markdown("<h1 style='font-size:40px;margin:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)

# compute (no st.cache_data to ensure fresh on every page open; if timeouts, add caching)
with st.spinner("Loading data from sheets..."):
    computed = compute_all()

dealers = computed.get("dealers", pd.DataFrame())
visits = computed.get("visits", pd.DataFrame())
clust_df = computed.get("clust_df", pd.DataFrame())
avail_df_merge = computed.get("avail_df_merge", pd.DataFrame())
areas_ordered = computed.get("areas_ordered", [])

# prepare BDE list excluding deleted-/trainer (already cleaned in data_preprocess but safe here)
bde_candidates = sorted(visits["employee_name"].dropna().astype(str).unique().tolist()) if not visits.empty else []
bde_select = st.selectbox("BDE Name", ["All"] + bde_candidates, index=0)

cols = st.columns([2,2,1])
with cols[0]:
    penetrated = st.multiselect("Dealer Activity", ['Not Active','Not Penetrated','Active'], default=['Not Penetrated','Active','Not Active'])
with cols[1]:
    potential = st.multiselect("Dealer Availability", ['Potential','Low Generation','Deficit'], default=['Potential','Low Generation','Deficit'])
with cols[2]:
    radius = st.slider("Radius (km)", 0, 50, 15)

# city & area filters
cities = sorted(avail_df_merge["city"].dropna().unique().tolist()) if not avail_df_merge.empty else []
area_default = areas_ordered if areas_ordered else []
areas_ui = st.multiselect("Area (ordered by dealer volume)", ["All"] + area_default, default=["All"])
if "All" in areas_ui:
    selected_areas = area_default
else:
    selected_areas = areas_ui

cities_ui = st.multiselect("City", ["All"] + cities, default=["All"])
if "All" in cities_ui:
    selected_cities = cities
else:
    selected_cities = cities_ui

brands = sorted(avail_df_merge["brand"].dropna().unique().tolist()) if not avail_df_merge.empty else []
brand_ui = st.multiselect("Brand", ["All"] + brands, default=["All"])
if "All" in brand_ui:
    selected_brands = brands
else:
    selected_brands = brand_ui

if st.button("Submit"):
    # pick avail candidates within any dist_center_N <= radius OR fallback to all
    df = avail_df_merge.copy() if not avail_df_merge.empty else pd.DataFrame()
    if df.empty:
        st.info("No dealer data available.")
        st.stop()

    dist_cols = [c for c in df.columns if c.startswith("dist_center_")]
    if dist_cols:
        mask = np.zeros(len(df), dtype=bool)
        for c in dist_cols:
            mask = mask | (pd.to_numeric(df[c], errors='coerce') <= radius)
        filtered = df[mask].copy()
    else:
        filtered = df.copy()

    # filter by name/area/city/brand/availability/tag
    if bde_select != "All":
        if "sales_name" in filtered.columns:
            filtered = filtered[filtered["sales_name"] == bde_select]
    if selected_areas:
        if "cluster" in filtered.columns:
            filtered = filtered[filtered["cluster"].astype(str).isin(selected_areas)]
    if selected_cities:
        if "city" in filtered.columns:
            filtered = filtered[filtered["city"].isin(selected_cities)]
    if selected_brands:
        if "brand" in filtered.columns:
            filtered = filtered[filtered["brand"].isin(selected_brands)]
    if potential:
        if "availability" in filtered.columns:
            filtered = filtered[filtered["availability"].isin(potential)]
    if penetrated:
        if "tag" in filtered.columns:
            filtered = filtered[filtered["tag"].isin(penetrated)]

    # join run-order info if exists (joined_dse, active_dse, nearest_end_date)
    # already merged in compute_availability if running_order sheet provided

    filtered = filtered.drop_duplicates(subset=["id_dealer_outlet"]) if "id_dealer_outlet" in filtered.columns else filtered.drop_duplicates()

    if filtered.empty:
        st.info("No dealers match the filters.")
        st.stop()

    # map center
    center_lat = float(filtered["latitude"].astype(float).mean())
    center_lon = float(filtered["longitude"].astype(float).mean())

    # color by tag
    def color_for_tag(t):
        if t == "Not Penetrated": return [131,201,255,200]
        if t == "Not Active": return [255,171,171,200]
        if t == "Active": return [255,43,43,200]
        return [200,200,200,200]

    filtered["color"] = filtered.get("tag", pd.Series(["Not Penetrated"]*len(filtered))).apply(lambda t: color_for_tag(t))
    dealer_records = filtered.to_dict(orient="records")
    center_records = clust_df.to_dict(orient="records") if not clust_df.empty else []

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
        tooltip={'text': "Brand: {brand}\nPenetration: {tag}"},
        layers=[
            pdk.Layer("ScatterplotLayer", data=dealer_records, get_position="[longitude,latitude]", get_radius=200, get_fill_color="color", pickable=True, auto_highlight=True),
            pdk.Layer("TextLayer", data=center_records, get_position="[longitude,latitude]", get_text="cluster", get_size=12, get_color=[0,100,0], get_text_anchor=String("middle"), get_alignment_baseline=String("center"))
        ]
    )
    st.pydeck_chart(deck)

    # Charts: Brand penetration bar and Potential sunburst
    st.markdown("### Dealer Penetration")
    if "tag" in filtered.columns and "brand" in filtered.columns:
        bar_src = filtered.groupby(["brand","tag"]).size().reset_index(name="Count")
        fig = px.bar(bar_src, x="brand", y="Count", color="tag")
        fig.update_layout(legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    if "availability" in filtered.columns and "brand" in filtered.columns:
        pot = filtered.groupby(["availability","brand"]).size().reset_index(name="Total Dealers")
        if not pot.empty:
            fig1 = px.sunburst(pot, path=['availability','brand'], values='Total Dealers')
            st.plotly_chart(fig1, use_container_width=True)

    # Table
    st.markdown("### Dealers Details")
    show_cols = []
    for c in ["name","brand","city","tag","joined_dse","active_dse","nearest_end_date","availability"]:
        if c in filtered.columns:
            show_cols.append(c)
    tbl = filtered[show_cols].rename(columns={
        "name":"Dealer Name",
        "brand":"Brand",
        "city":"City",
        "tag":"Activity",
        "joined_dse":"Total Joined DSE",
        "active_dse":"Total Active DSE",
        "nearest_end_date":"Nearest Package End Date",
        "availability":"Availability"
    }).drop_duplicates().reset_index(drop=True)
    st.dataframe(tbl)
