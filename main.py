import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pydeck as pdk
from pydeck.types import String
from data_preprocess import compute_all

st.set_page_config(page_title="Dealer Penetration Dashboard", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1rem;max-width:1200px}
h1{margin:0 0 8px 0;font-size:34px}
.stButton>button{border-radius:10px;height:40px;padding:0 16px}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>Dealer Penetration Dashboard</h1>", unsafe_allow_html=True)

with st.container(border=True):
    c1,c2,c3,c4 = st.columns([1.1,1.1,1.1,0.8])
    with c1:
        radius = st.slider("Radius (km)", 0, 50, 15)
    with c2:
        penetrated = st.multiselect("Dealer Activity", ["All","Not Active","Not Penetrated","Active"], default=["All"])
    with c3:
        potential = st.multiselect("Dealer Availability", ["All","Potential","Low Generation","Deficit"], default=["All"])
    with c4:
        go = st.button("Apply", use_container_width=True)

if not go:
    st.stop()

with st.spinner("Loading..."):
    computed = compute_all()

dealers = computed["dealers"]
visits = computed["visits"]
sum_df = computed["sum_df"]
df = computed["avail_df_merge"]
clust_df = computed["clust_df"]

if dealers.empty or df.empty:
    st.info("Data tidak tersedia.")
    st.stop()

area_pool = ["All"] + sorted(df.get("cluster", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
city_pool = ["All"] + sorted(df.get("city", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
brand_pool = ["All"] + sorted(df.get("brand", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())

with st.container(border=True):
    a1,a2,a3 = st.columns(3)
    with a1:
        area_pick = st.multiselect("Area", area_pool, default=["All"])
    with a2:
        city_pick = st.multiselect("City", city_pool, default=["All"])
    with a3:
        brand = st.multiselect("Brand", brand_pool, default=["All"])

df["joined_dse"] = pd.to_numeric(df.get("joined_dse", 0), errors="coerce").fillna(0)
df["active_dse"] = pd.to_numeric(df.get("active_dse", 0), errors="coerce").fillna(0)
df["tag"] = np.where((df["joined_dse"]==0) & (df["active_dse"]==0), "Not Penetrated", df.get("tag","Not Active"))
df["availability"] = df.get("availability","Potential")
df["cluster"] = df.get("cluster","").astype(str)

if area_pick and "All" not in area_pick:
    df = df[df["cluster"].isin(area_pick)]
if city_pick and "All" not in city_pick:
    df = df[df["city"].astype(str).isin(city_pick)]
if brand and "All" not in brand:
    df = df[df["brand"].astype(str).isin(brand)]
if penetrated and "All" not in penetrated:
    df = df[df["tag"].astype(str).isin(penetrated)]
if potential and "All" not in potential:
    df = df[df["availability"].astype(str).isin(potential)]

if df.empty:
    st.info("Tidak ada dealer yang memenuhi filter.")
    st.stop()

center_lon = float(pd.to_numeric(df["longitude"], errors="coerce").mean())
center_lat = float(pd.to_numeric(df["latitude"], errors="coerce").mean())

clusters = clust_df.copy()
if not clusters.empty and not sum_df.empty:
    visits_cnt = sum_df.groupby(["sales_name","cluster"], as_index=False)["cluster"].count().rename(columns={"cluster":"count_visit"})
    clusters = clusters.merge(visits_cnt, on=["sales_name","cluster"], how="left")
    total_vis = max(clusters["count_visit"].fillna(0).sum(), 1)
    clusters["size"] = clusters["count_visit"].fillna(0)/total_vis*9000
    clusters["word"] = "Area " + (clusters["cluster"].astype(int)+1).astype(str) + "\nCount Visit: " + clusters["count_visit"].fillna(0).astype(int).astype(str)
else:
    clusters = pd.DataFrame([{"latitude":center_lat,"longitude":center_lon,"size":5000,"word":"Area"}])

st.markdown("### Penetration Map")
deck = pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
    tooltip={"text":"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nActivity: {tag}"},
    layers=[
        pdk.Layer("TextLayer", data=clusters, get_position="[longitude,latitude]", get_text="word", get_size=12, get_color=[0,1000,0], get_text_anchor=String("middle"), get_alignment_baseline=String("center")),
        pdk.Layer("ScatterplotLayer", data=df, get_position="[longitude,latitude]", get_radius=200, get_fill_color=[21,255,87,200], pickable=True, auto_highlight=True),
        pdk.Layer("ScatterplotLayer", data=clusters, get_position="[longitude,latitude]", get_radius="size", get_fill_color=[200,30,0,90]),
    ],
)
st.pydeck_chart(deck, use_container_width=True)

k1,k2,k3,k4 = st.columns(4)
with k1:
    st.metric("Dealers", int(df["id_dealer_outlet"].nunique()))
with k2:
    st.metric("Active Dealers", int((df["active_dse"].fillna(0)>0).sum()))
with k3:
    st.metric("Active DSE", int(df["active_dse"].fillna(0).sum()))
with k4:
    rev_mtd = pd.to_numeric(df.get("revenue_mtd", 0), errors="coerce").fillna(0).sum()
    st.metric("Revenue MTD", f"{rev_mtd:,.0f}")

st.markdown("### Insights")
bar_src = df.groupby(["brand","tag"], as_index=False)["id_dealer_outlet"].count().rename(columns={"id_dealer_outlet":"Count Dealers","brand":"Brand","tag":"Status"})
fig = px.bar(bar_src, x="Brand", y="Count Dealers", color="Status", barmode="group")
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

pie_src = df.groupby(["availability"], as_index=False)["id_dealer_outlet"].count().rename(columns={"id_dealer_outlet":"Total"})
pie = px.pie(pie_src, names="availability", values="Total", hole=0.35)
st.plotly_chart(pie, use_container_width=True)

st.markdown("### Dealers Detail")
show_cols = ["cluster","brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability","revenue_total","revenue_mtd","revenue_last_30d"]
show_cols = [c for c in show_cols if c in df.columns]
df_table = df[show_cols].rename(columns={"cluster":"Area","brand":"Brand","name":"Dealer Name","city":"City","tag":"Activity","joined_dse":"Total Joined DSE","active_dse":"Total Active DSE","nearest_end_date":"Nearest Package End Date","availability":"Availability","revenue_total":"Revenue Total","revenue_mtd":"Revenue MTD","revenue_last_30d":"Revenue 30D"}).drop_duplicates().reset_index(drop=True)
st.dataframe(df_table, use_container_width=True)
