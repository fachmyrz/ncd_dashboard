# main.py
import streamlit as st
from PIL import Image
import pandas as pd
import pydeck as pdk
from pydeck.types import String
import plotly.express as px

from data_preprocess import compute_all
import data_load

# page config
try:
    icon = Image.open("assets/favicon.png")
    st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=icon, layout="wide")
except:
    st.set_page_config(page_title="Dealer Penetration Dashboard", layout="wide")

st.markdown("<h1 style='font-size:40px;margin:0'>Dealer Penetration Dashboard</h1>", unsafe_allow_html=True)
col_refresh = st.sidebar.button("Refresh Data (force)")

# Force reload sheet cache if requested
if col_refresh:
    # clear caches in streamlit (call underlying functions to clear)
    st.cache_data.clear()

# load computed dataset (this will be fast on repeated opens because of cache)
with st.spinner("Loading and computing — this may take a few seconds on first load..."):
    computed = compute_all()

dealers = computed.get("dealers", pd.DataFrame())
visits = computed.get("visits", pd.DataFrame())
avail = computed.get("avail", pd.DataFrame())
clust_df = computed.get("clust_df", pd.DataFrame())
area_order = computed.get("area_order", [])

# quick checks
if dealers.empty:
    st.warning("No dealer data — check your sheet ids / sheet names in Streamlit secrets.")
    st.stop()

# Filters
bde_list = sorted(visits["employee_name"].dropna().astype(str).unique().tolist()) if not visits.empty else []
bde = st.selectbox("BDE Name", ["All"] + bde_list, index=0)

areas = ["All"] + area_order
area_sel = st.multiselect("Area", areas, default=["All"])
if "All" in area_sel:
    selected_areas = area_order
else:
    selected_areas = area_sel

cities = sorted(avail["city"].dropna().unique().tolist()) if not avail.empty else []
city_sel = st.multiselect("City", ["All"] + cities, default=["All"])
selected_cities = cities if "All" in city_sel else city_sel

brands = sorted(avail["brand"].dropna().unique().tolist()) if not avail.empty else []
brand_sel = st.multiselect("Brand", ["All"] + brands, default=["All"])
selected_brands = brands if "All" in brand_sel else brand_sel

penetrated = st.multiselect("Dealer Activity", ['Not Active','Not Penetrated','Active'], default=['Not Penetrated','Active','Not Active'])
availability = st.multiselect("Dealer Availability", ['Potential','Low Generation','Deficit'], default=['Potential','Low Generation','Deficit'])
radius = st.slider("Radius (km) for area filter", 0, 50, 15)

submit = st.button("Submit")

def filter_avail(avail_df):
    df = avail_df.copy()
    # filter by dist columns
    dist_cols = [c for c in df.columns if c.startswith("dist_center_")]
    if dist_cols:
        mask_any = pd.Series(False, index=df.index)
        for c in dist_cols:
            mask_any = mask_any | (pd.to_numeric(df[c], errors='coerce') <= radius)
        df = df[mask_any]
    # BDE
    if bde != "All" and "sales_name" in df.columns:
        df = df[df["sales_name"] == bde]
    if selected_areas and "cluster" in df.columns:
        df = df[df["cluster"].astype(str).isin(selected_areas)]
    if selected_cities and "city" in df.columns:
        df = df[df["city"].isin(selected_cities)]
    if selected_brands and "brand" in df.columns:
        df = df[df["brand"].isin(selected_brands)]
    if availability and "availability" in df.columns:
        df = df[df["availability"].isin(availability)]
    if penetrated and "tag" in df.columns:
        df = df[df["tag"].isin(penetrated)]
    return df

if submit:
    with st.spinner("Applying filters..."):
        filtered = filter_avail(avail)
        if filtered.empty:
            st.info("No dealers match the selected filters.")
            st.stop()

        # center map
        center_lat = float(filtered["latitude"].astype(float).mean())
        center_lon = float(filtered["longitude"].astype(float).mean())

        # color mapping
        def color_for_tag(t):
            if t == "Not Penetrated": return [131,201,255,200]
            if t == "Not Active": return [255,171,171,200]
            if t == "Active": return [255,43,43,200]
            return [200,200,200,200]

        filtered["color"] = filtered.get("tag", pd.Series(["Not Penetrated"]*len(filtered))).apply(color_for_tag)

        # Performance: limit plotted points by default (set to all via toggle)
        plot_all = st.checkbox("Plot all dealers (may be slower)", value=False)
        plot_df = filtered if plot_all else filtered.head(200)

        deck = pdk.Deck(
            map_style="LIGHT",
            initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=45),
            tooltip={"text":"Dealer: {name}\nBrand: {brand}\nActivity: {tag}"},
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=plot_df.to_dict(orient="records"),
                    get_position='[longitude, latitude]',
                    get_fill_color="color",
                    get_radius=200,
                    pickable=True,
                    auto_highlight=True
                ),
            ]
        )
        st.pydeck_chart(deck)

        # Bar chart: brand vs activity
        if "brand" in filtered.columns and "tag" in filtered.columns:
            bar_src = filtered.groupby(["brand","tag"]).size().reset_index(name="count")
            fig = px.bar(bar_src, x="brand", y="count", color="tag", title="Brand penetration")
            st.plotly_chart(fig, use_container_width=True)

        # Sunburst: availability x brand
        if "availability" in filtered.columns:
            pot = filtered.groupby(["availability","brand"]).size().reset_index(name="count")
            if not pot.empty:
                fig2 = px.sunburst(pot, path=["availability","brand"], values="count", title="Potential dealers")
                st.plotly_chart(fig2, use_container_width=True)

        # Table
        show_cols = []
        for c in ["name","brand","city","tag","joined_dse","active_dse","nearest_end_date","availability"]:
            if c in filtered.columns:
                show_cols.append(c)
        df_show = filtered[show_cols].rename(columns={
            "name":"Dealer Name",
            "brand":"Brand",
            "city":"City",
            "tag":"Activity",
            "joined_dse":"Total Joined DSE",
            "active_dse":"Total Active DSE",
            "nearest_end_date":"Nearest Package End Date",
            "availability":"Availability"
        }).drop_duplicates().reset_index(drop=True)
        st.dataframe(df_show)
