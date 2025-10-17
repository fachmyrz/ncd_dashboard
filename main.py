import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pydeck as pdk
from pydeck.types import String
from data_preprocess import compute_base, cluster_one_bde

st.set_page_config(page_title="Dealer Penetration Dashboard", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1180px}
.stButton>button{border-radius:10px;height:42px;padding:0 20px}
.stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"]{border-radius:12px}
h1{margin:0 0 6px 0;font-size:40px}
.section{margin:8px 0 4px 0}
hr{margin:8px 0}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>Dealer Penetration Dashboard</h1>", unsafe_allow_html=True)

dealers, visits, avail = compute_base()
if dealers.empty or avail.empty:
    st.info("Data kosong. Cek konfigurasi Sheet di secrets.")
    st.stop()

bde_pool = ["All"] + sorted(visits["employee_name"].dropna().astype(str).unique().tolist()) if "employee_name" in visits.columns else ["All"]
area_pool = sorted(avail["cluster"].dropna().astype(str).unique().tolist())
city_all = sorted(avail["city"].dropna().astype(str).unique().tolist())
brand_all = sorted(avail["brand"].dropna().astype(str).unique().tolist())

r1c1, r1c2, r1c3, r1c4 = st.columns([1.1,1.1,1.1,0.8])
with r1c1:
    bde = st.selectbox("BDE Name", bde_pool, index=0)
with r1c2:
    area_pick = st.multiselect("Area", ["All"]+area_pool, default=["All"])
    if "All" in area_pick or not area_pick:
        area_pick = area_pool
with r1c3:
    city_pool = sorted(avail[avail["cluster"].astype(str).isin(area_pick)]["city"].dropna().astype(str).unique().tolist())
    city_pick = st.multiselect("City", ["All"]+city_pool, default=["All"])
    if "All" in city_pick or not city_pick:
        city_pick = city_pool
with r1c4:
    radius = st.slider("Radius (km)", 0, 50, 15, disabled=(bde=="All"))

r2c1, r2c2 = st.columns([1.1,1.1])
with r2c1:
    brand_pool = sorted(avail[(avail["cluster"].astype(str).isin(area_pick)) & (avail["city"].astype(str).isin(city_pick))]["brand"].dropna().astype(str).unique().tolist())
    brand = st.multiselect("Brand", ["All"]+brand_pool, default=["All"])
    if "All" in brand or not brand:
        brand = brand_pool
with r2c2:
    col_a, col_b = st.columns([1,1])
    with col_a:
        penetrated = st.multiselect("Dealer Activity", ["All","Not Active","Not Penetrated","Active"], default=["All"])
        if "All" in penetrated or not penetrated:
            penetrated = ["Not Active","Not Penetrated","Active"]
    with col_b:
        potential = st.multiselect("Dealer Availability", ["All","Potential","Low Generation","Deficit"], default=["All"])
        if "All" in potential or not potential:
            potential = ["Potential","Low Generation","Deficit"]

btn_col = st.columns([0.2,0.8,0.2])[0]
go = btn_col.button("Apply Filters", use_container_width=False)

if not go:
    st.stop()

with st.spinner("Mengolah data..."):
    if bde != "All":
        sum_df, avail_dist, clust_df = cluster_one_bde(visits, dealers, bde)
    else:
        sum_df, avail_dist, clust_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df = avail.copy()
if not avail_dist.empty:
    dist_cols = [c for c in avail_dist.columns if str(c).startswith("dist_center_")]
    if dist_cols:
        vals = avail_dist[dist_cols].apply(pd.to_numeric, errors="coerce")
        mvals = vals.min(axis=1)
        for c in dist_cols:
            vc = pd.to_numeric(avail_dist[c], errors="coerce")
            mask = np.isfinite(vc) & np.isfinite(mvals) & np.isclose(vc, mvals, atol=1e-9)
            avail_dist[c] = np.where(mask, vc, np.nan)
        try:
            df = df.merge(avail_dist[["id_dealer_outlet"]+dist_cols], on="id_dealer_outlet", how="left")
        except Exception:
            pass

df["nearest_end_date"] = pd.to_datetime(df["nearest_end_date"], errors="coerce")
df["status_days_to_expire"] = (df["nearest_end_date"] - pd.Timestamp.today().normalize()).dt.days
df["will_expire_30d"] = np.where(df["status_days_to_expire"].between(0,30, inclusive="both"), True, False)
df["tag"] = np.where((pd.to_numeric(df["joined_dse"], errors="coerce").fillna(0)==0) & (pd.to_numeric(df["active_dse"], errors="coerce").fillna(0)==0), "Not Penetrated", df["tag"].fillna("Not Active"))
df["availability"] = df["availability"].fillna("Potential")

df = df[df["cluster"].astype(str).isin(area_pick)]
df = df[df["city"].astype(str).isin(city_pick)]
df = df[df["brand"].astype(str).isin(brand)]
df = df[df["tag"].astype(str).isin(penetrated)]
df = df[df["availability"].astype(str).isin(potential)]

if bde != "All":
    picks = []
    if not clust_df.empty:
        for i in sorted(clust_df["cluster"].astype(int).unique().tolist()):
            col = f"dist_center_{i}"
            if col in df.columns:
                part = df[(pd.to_numeric(df[col], errors="coerce").notna()) & (pd.to_numeric(df[col], errors="coerce") <= radius)].copy()
                if not part.empty:
                    part["cluster_labels"] = int(i)
                    part["sales_name"] = bde
                    picks.append(part)
    df_pick = pd.concat(picks, ignore_index=True) if picks else df.copy()
else:
    df_pick = df.copy()

if df_pick.empty:
    st.info("Tidak ada dealer yang cocok dengan filter.")
    st.stop()

df_pick["joined_dse"] = pd.to_numeric(df_pick["joined_dse"], errors="coerce").fillna(0)
df_pick["active_dse"] = pd.to_numeric(df_pick["active_dse"], errors="coerce").fillna(0)
df_pick["engagement_bucket"] = np.select([(df_pick["active_dse"]>0),(df_pick["joined_dse"]>0) & (df_pick["active_dse"]==0),(df_pick["joined_dse"]==0) & (df_pick["active_dse"]==0)],["Active","Not Active","Not Penetrated"],default="Not Penetrated")
center_lat = float(pd.to_numeric(df_pick["latitude"], errors="coerce").mean())
center_lon = float(pd.to_numeric(df_pick["longitude"], errors="coerce").mean())
color_map = {"Active":[21,255,87,200],"Not Active":[255,171,171,200],"Not Penetrated":[131,201,255,200]}
df_pick["color"] = df_pick["engagement_bucket"].map(color_map).apply(lambda x: x if isinstance(x, list) else [200,200,200,180])

clusters = clust_df.copy()
if not clusters.empty:
    visits_cnt = sum_df.groupby(["sales_name","cluster"], as_index=False)["cluster"].count().rename(columns={"cluster":"count_visit"})
    clusters = clusters.merge(visits_cnt, on=["sales_name","cluster"], how="left")
    total_vis = max(float(clusters["count_visit"].fillna(0).sum()), 1.0)
    clusters["size"] = clusters["count_visit"].fillna(0)/total_vis*9000
    clusters["word"] = "Area " + (pd.to_numeric(clusters["cluster"], errors="coerce").fillna(0).astype(int)+1).astype(str) + "\nCount Visit: " + clusters["count_visit"].fillna(0).astype(int).astype(str)
else:
    clusters = pd.DataFrame([{"latitude":center_lat,"longitude":center_lon,"size":5000,"word":"Area"}])

st.markdown("<div class='section'></div>", unsafe_allow_html=True)
deck = pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
    tooltip={"text":"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nActivity: {engagement_bucket}"},
    layers=[
        pdk.Layer("TextLayer", data=clusters, get_position="[longitude,latitude]", get_text="word", get_size=12, get_color=[0,128,0], get_text_anchor=String("middle"), get_alignment_baseline=String("center")),
        pdk.Layer("ScatterplotLayer", data=df_pick, get_position="[longitude,latitude]", get_radius=200, get_fill_color="color", pickable=True, auto_highlight=True),
        pdk.Layer("ScatterplotLayer", data=clusters, get_position="[longitude,latitude]", get_radius="size", get_fill_color=[200,30,0,90]),
    ],
)
st.pydeck_chart(deck)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Dealers", int(pd.to_numeric(df_pick["id_dealer_outlet"], errors="coerce").nunique()))
with kpi2:
    st.metric("Active Dealers", int((df_pick["active_dse"]>0).sum()))
with kpi3:
    st.metric("Active DSE", int(df_pick["active_dse"].sum()))
with kpi4:
    if "visit_datetime" in visits.columns and "matched_dealer_id" in visits.columns:
        wv = visits.dropna(subset=["visit_datetime"]).copy()
        wv["week"] = pd.to_datetime(wv["visit_datetime"], errors="coerce").dt.to_period("W").astype(str)
        avg_week = wv.groupby("matched_dealer_id")["week"].nunique().mean()
        st.metric("Avg Weekly Visits", round(float(avg_week if pd.notna(avg_week) else 0),2))
    else:
        st.metric("Avg Weekly Visits", 0)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<h3 style='margin:4px 0'>Dealers Detail per Area</h3>", unsafe_allow_html=True)

def _area_view(area_label):
    dfa = df_pick.copy()
    if "cluster_labels" in dfa.columns:
        dfa["area_tag"] = dfa["cluster_labels"].astype(int) + 1
        dfa["area_tag_word"] = "Area " + dfa["area_tag"].astype(str)
        dfa = dfa[dfa["area_tag_word"]==area_label]
    st.markdown(f"**{area_label}** • {len(dfa)} dealers")
    s1 = dfa.groupby(["brand","engagement_bucket"], as_index=False)["id_dealer_outlet"].count().rename(columns={"id_dealer_outlet":"Count Dealers","brand":"Brand","engagement_bucket":"Status"})
    s2 = dfa.groupby(["availability","brand"], as_index=False)["id_dealer_outlet"].count().rename(columns={"id_dealer_outlet":"Total","availability":"Availability","brand":"Brand"})
    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.bar(s1, x="Brand", y="Count Dealers", color="Status", barmode="group")
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sun = px.sunburst(s2, path=["Availability","Brand"], values="Total")
        st.plotly_chart(sun, use_container_width=True)
    show_cols = ["cluster","brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability","will_expire_30d"]
    tbl = dfa[show_cols].rename(columns={"cluster":"Area","brand":"Brand","name":"Dealer Name","city":"City","tag":"Activity","joined_dse":"Total Joined DSE","active_dse":"Total Active DSE","nearest_end_date":"Nearest Package End Date","availability":"Availability","will_expire_30d":"Expire ≤30d"}).drop_duplicates().reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True)

if bde != "All" and not clusters.empty:
    tabs = ["Area "+str(i+1) for i in sorted(clusters["cluster"].astype(int).unique().tolist())]
    t_objs = st.tabs(tabs)
    for t, lbl in zip(t_objs, tabs):
        with t:
            _area_view(lbl)
else:
    _area_view("Area")
