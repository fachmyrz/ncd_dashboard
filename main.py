import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pydeck as pdk
from pydeck.types import String
from PIL import Image
from data_load import get_sheets, clear_cache
from data_preprocess import compute_all

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=Image.open("assets/favicon.png") if "assets/favicon.png" else None, layout="wide")
st.sidebar.button("Refresh Data (force)", on_click=clear_cache)
st.markdown("<h1 style='font-size:40px;margin:0'>Dealer Penetration Dashboard</h1>", unsafe_allow_html=True)

computed = compute_all()
dealers = computed["dealers"]
visits = computed["visits"]
sum_df = computed["sum_df"]
clust_df = computed["clust_df"]
avail_df_merge = computed["avail_df_merge"]

if dealers.empty:
    st.info("No dealer data — check your sheet ids / sheet names in Streamlit secrets.")
    st.stop()

bde_base = visits[["employee_name","nik","divisi"]].drop_duplicates() if not visits.empty else pd.DataFrame(columns=["employee_name","nik","divisi"])
bde_list = ["All"] + sorted(bde_base["employee_name"].dropna().astype(str).unique().tolist())

ld = get_sheets()["location_detail"].rename(columns={"City":"city","Cluster":"cluster"})
area_order = dealers.merge(ld[["city","cluster"]], on="city", how="left").groupby("cluster", as_index=False)["id_dealer_outlet"].count().sort_values("id_dealer_outlet", ascending=False)["cluster"].astype(str).dropna().tolist()
area_choices = ["All"] + area_order

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,0.8])
    with c1:
        bde = st.selectbox("BDE Name", bde_list, index=0)
    with c2:
        area_pick = st.multiselect("Area (Cluster)", area_choices, default=["All"])
        if "All" in area_pick:
            area_pick = area_choices[1:] if len(area_choices)>1 else []
    with c3:
        city_pool = dealers["city"].dropna().unique().tolist()
        city_pick = st.multiselect("City", ["All"] + city_pool, default=["All"])
        if "All" in city_pick:
            city_pick = city_pool
    with c4:
        radius = st.slider("Radius (km)", 0, 50, 15)

c5, c6, c7, c8 = st.columns([1.2,1.2,1.2,0.8])
with c5:
    penetrated = st.multiselect("Dealer Activity", ["All","Not Active","Not Penetrated","Active"], default=["All"])
with c6:
    potential = st.multiselect("Dealer Availability", ["All","Potential","Low Generation","Deficit"], default=["All"])
with c7:
    brand_pool = dealers["brand"].dropna().unique().tolist()
    brand = st.multiselect("Brand", ["All"] + brand_pool, default=["All"])
with c8:
    show_heatmap = st.toggle("Visits Heatmap", value=False)

btn = st.button("Apply Filters", type="primary", use_container_width=True)

if not btn:
    st.info("Set filters and click Apply Filters.")
    st.stop()

df = avail_df_merge.copy()
df["nearest_end_date"] = pd.to_datetime(df["nearest_end_date"], errors="coerce")
df["status_days_to_expire"] = (df["nearest_end_date"] - pd.Timestamp.today().normalize()).dt.days
df["will_expire_30d"] = np.where(df["status_days_to_expire"].between(0,30, inclusive="both"), True, False)
df["tag"] = np.where((df["joined_dse"].fillna(0)==0) & (df["active_dse"].fillna(0)==0), "Not Penetrated", df["tag"].fillna("Not Active"))
df["availability"] = df["availability"].fillna("Potential")
df["cluster"] = df["cluster"].astype(str)

if bde != "All" and not sum_df.empty:
    picks = []
    n_clusters = len(sum_df[sum_df["sales_name"]==bde]["cluster"].unique())
    for i in range(n_clusters):
        col = f"dist_center_{i}"
        if col in df.columns:
            tp = df[(df[col].notna()) & (df[col] <= radius)].copy()
            tp["cluster_labels"] = i
            tp["sales_name"] = bde
            picks.append(tp)
    df_pick = pd.concat(picks, ignore_index=True) if picks else df.copy()
else:
    df_pick = df.copy()

if area_pick:
    df_pick = df_pick[df_pick["cluster"].astype(str).isin(area_pick)]
if penetrated and "All" not in penetrated:
    df_pick = df_pick[df_pick["tag"].isin(penetrated)]
if potential and "All" not in potential:
    df_pick = df_pick[df_pick["availability"].isin(potential)]
if city_pick:
    df_pick = df_pick[df_pick["city"].isin(city_pick)]
if brand and "All" not in brand:
    df_pick = df_pick[df_pick["brand"].isin(brand)]

if df_pick.empty:
    st.info("No dealers match your filters.")
    st.stop()

df_pick["joined_dse"] = df_pick["joined_dse"].fillna(0)
df_pick["active_dse"] = df_pick["active_dse"].fillna(0)
df_pick["engagement_bucket"] = np.select(
    [
        (df_pick["active_dse"]>0),
        (df_pick["joined_dse"]>0) & (df_pick["active_dse"]==0),
        (df_pick["joined_dse"]==0) & (df_pick["active_dse"]==0),
    ],
    ["Active","Not Active","Not Penetrated"],
    default="Not Penetrated"
)

center_lon = float(df_pick["longitude"].mean())
center_lat = float(df_pick["latitude"].mean())
color_map = {"Active":[21,255,87,200],"Not Active":[255,171,171,200],"Not Penetrated":[131,201,255,200]}
df_pick["color"] = df_pick["engagement_bucket"].map(color_map).apply(lambda x: x if isinstance(x, list) else [200,200,200,180])

clusters = clust_df.copy()
if not clusters.empty and bde != "All":
    clusters = clusters[clusters["sales_name"]==bde]
if not clusters.empty and not sum_df.empty:
    visits_cnt = sum_df.groupby(["sales_name","cluster"], as_index=False)["cluster"].count().rename(columns={"cluster":"count_visit"})
    clusters = clusters.merge(visits_cnt, on=["sales_name","cluster"], how="left")
    total_vis = max(clusters["count_visit"].fillna(0).sum(), 1)
    clusters["size"] = clusters["count_visit"].fillna(0)/total_vis*9000
    clusters["word"] = "Area " + (clusters["cluster"].astype(int)+1).astype(str) + "\nCount Visit: " + clusters["count_visit"].fillna(0).astype(int).astype(str)
else:
    clusters = pd.DataFrame([{"latitude":center_lat,"longitude":center_lon,"size":5000,"word":"Area"}])

heat_layer = []
if show_heatmap and not visits.empty:
    v = visits.dropna(subset=["lat","long"]).rename(columns={"lat":"latitude","long":"longitude"})
    heat_layer = [
        pdk.Layer(
            "HeatmapLayer",
            data=v,
            get_position="[longitude, latitude]",
            aggregation="MEAN",
            threshold=0.1,
            get_weight=1,
            radius_pixels=40,
        )
    ]

deck_layers = [
    pdk.Layer(
        "TextLayer",
        data=clusters,
        get_position="[longitude,latitude]",
        get_text="word",
        get_size=12,
        get_color=[0,128,0],
        get_text_anchor=String("middle"),
        get_alignment_baseline=String("center"),
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=df_pick,
        get_position="[longitude,latitude]",
        get_radius=200,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=clusters,
        get_position="[longitude,latitude]",
        get_radius="size",
        get_fill_color=[200,30,0,90],
    ),
] + heat_layer

st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50), tooltip={"text":"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nActivity: {engagement_bucket}"}, layers=deck_layers))

with st.expander("Legend", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown("🟩 <b>Active</b>", unsafe_allow_html=True)
    lc2.markdown("🟥 <b>Not Active</b>", unsafe_allow_html=True)
    lc3.markdown("🟦 <b>Not Penetrated</b>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Dealers", int(df_pick["id_dealer_outlet"].nunique()))
with kpi2:
    st.metric("Active Dealers", int((df_pick["active_dse"].fillna(0)>0).sum()))
with kpi3:
    st.metric("Active DSE", int(df_pick["active_dse"].fillna(0).sum()))
with kpi4:
    st.metric("Expiring ≤ 30 days", int(df_pick["will_expire_30d"].sum()))

area_col = "cluster"
df_pick["area_tag_word"] = "Area " + df_pick.get("cluster_labels", pd.Series(0, index=df_pick.index)).fillna(0).astype(int).add(1).astype(str)
df_pick["area"] = df_pick[area_col].astype(str)

st.markdown("<h2 style='font-size:24px;margin:8px 0'>Insights</h2>", unsafe_allow_html=True)
bar_src = df_pick.groupby(["brand","engagement_bucket"], as_index=False)["id_dealer_outlet"].count().rename(columns={"id_dealer_outlet":"Count Dealers","brand":"Brand","engagement_bucket":"Status"}).sort_values(["Brand","Status"])
fig_bar = px.bar(bar_src, x="Brand", y="Count Dealers", color="Status", barmode="group")
fig_bar.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_bar, use_container_width=True, key="bar_brand_status")

pie_src = df_pick.groupby(["availability"], as_index=False)["id_dealer_outlet"].count().rename(columns={"id_dealer_outlet":"Total"})
fig_pie = px.pie(pie_src, names="availability", values="Total", hole=0.35)
st.plotly_chart(fig_pie, use_container_width=True, key="pie_avail")

st.markdown("<h2 style='font-size:24px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
show_cols = ["brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability","will_expire_30d"]
df_table = df_pick[show_cols].rename(columns={"brand":"Brand","name":"Dealer Name","city":"City","tag":"Activity","joined_dse":"Total Joined DSE","active_dse":"Total Active DSE","nearest_end_date":"Nearest Package End Date","availability":"Availability","will_expire_30d":"Expire ≤30d"}).drop_duplicates().reset_index(drop=True)
st.dataframe(df_table, use_container_width=True, hide_index=True)
csv = df_table.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download table (CSV)", data=csv, file_name="dealers_detail.csv", mime="text/csv")
