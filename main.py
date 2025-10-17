import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pydeck as pdk
from pydeck.types import String
from PIL import Image
from data_load import get_sheets
from data_preprocess import clean_dealers, clean_visits, compute_all

def _safe_icon():
    try:
        return Image.open("assets/favicon.png")
    except Exception:
        return None

def _ensure_cols(df, required):
    for c, default in required.items():
        if c not in df.columns:
            df[c] = default
    return df

def _coerce_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _finite_mask(df, lat_col="latitude", lon_col="longitude"):
    if lat_col not in df.columns or lon_col not in df.columns:
        return df.iloc[0:0]
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    m = lat.between(-90, 90) & lon.between(-180, 180) & lat.notna() & lon.notna()
    return df[m]

def _center_of(df, lat_col="latitude", lon_col="longitude"):
    df = _finite_mask(df, lat_col, lon_col)
    if df.empty:
        return -6.2000, 106.8167
    return float(df[lat_col].mean()), float(df[lon_col].mean())

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=_safe_icon(), layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem}
h1{letter-spacing:.2px}
[data-testid="stMetricValue"]{font-weight:700}
div[role="radiogroup"] label{padding:4px 8px;border-radius:8px}
.stMultiSelect [data-baseweb="select"]{border-radius:12px}
.stSelectbox [data-baseweb="select"]{border-radius:12px}
button[kind="primary"]{border-radius:14px}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='font-size:42px;margin:0 0 8px 0'>Dealer Penetration Dashboard</h1>", unsafe_allow_html=True)

sheets = get_sheets()
dealers_raw = sheets.get("dealers", pd.DataFrame())
visits_raw = sheets.get("visits", pd.DataFrame())
location_detail = sheets.get("location_detail", pd.DataFrame()).rename(columns={"City":"city","Cluster":"cluster"})
dealers_light = clean_dealers(dealers_raw) if not dealers_raw.empty else pd.DataFrame()
visits_light = clean_visits(visits_raw) if not visits_raw.empty else pd.DataFrame()
dealers_light = dealers_light.merge(location_detail[["city","cluster"]], on="city", how="left") if not dealers_light.empty and not location_detail.empty else dealers_light

if dealers_light.empty:
    st.info("No dealer data — check your Google Sheet IDs/tabs in secrets.")
    st.stop()

dealers_light = _ensure_cols(dealers_light.copy(), {"id_dealer_outlet": pd.NA, "brand": "", "city": "", "cluster": "", "name": "", "latitude": np.nan, "longitude": np.nan})
dealers_light = _coerce_float(dealers_light, ["latitude","longitude"])
dealers_light = _finite_mask(dealers_light, "latitude", "longitude")

bde_list = ["All"] + sorted(visits_light["employee_name"].dropna().astype(str).unique().tolist()) if "employee_name" in visits_light.columns else ["All"]
area_all = sorted([str(x) for x in dealers_light["cluster"].dropna().astype(str).unique().tolist() if x])
city_all = sorted([c for c in dealers_light["city"].dropna().astype(str).unique().tolist() if c])
brand_all = sorted([b for b in dealers_light["brand"].dropna().astype(str).unique().tolist() if b])

with st.container():
    r1c1, r1c2, r1c3, r1c4 = st.columns([1.1,1.1,1.1,0.7])
    with r1c1:
        bde = st.selectbox("BDE Name", bde_list, index=0)
    with r1c2:
        area_pick = st.multiselect("Area", ["All"] + area_all, default=["All"])
    with r1c3:
        if "All" in area_pick or not area_pick:
            cities_pool = city_all
        else:
            cities_pool = sorted(dealers_light[dealers_light["cluster"].astype(str).isin(area_pick)]["city"].dropna().astype(str).unique().tolist())
        city_pick = st.multiselect("City", ["All"] + cities_pool, default=["All"])
    with r1c4:
        radius = st.slider("Radius (km)", 0, 50, 15, disabled=(bde=="All"))

r2c1, r2c2 = st.columns([1.1,1.1])
with r2c1:
    if "All" in area_pick or not area_pick:
        brands_pool = brand_all
    else:
        brands_pool = sorted(dealers_light[dealers_light["cluster"].astype(str).isin(area_pick)]["brand"].dropna().astype(str).unique().tolist())
    brand_sel = st.multiselect("Brand", ["All"] + brands_pool, default=["All"])
with r2c2:
    st.write("")

r3c1, r3c2 = st.columns([1.1,1.1])
with r3c1:
    penetrated = st.multiselect("Dealer Activity", ["All","Not Active","Not Penetrated","Active"], default=["All"])
with r3c2:
    potential = st.multiselect("Dealer Availability", ["All","Potential","Low Generation","Deficit"], default=["All"])

btn = st.button("Apply Filters", type="primary", use_container_width=True)
if not btn:
    st.stop()

with st.status("Loading data and computing clusters…", expanded=False) as s:
    try:
        computed = compute_all(bde_filter=bde)
        s.update(label="Computation done.")
    except Exception as e:
        s.update(label="Failed to compute data.")
        st.exception(e)
        st.stop()

dealers = computed.get("dealers", pd.DataFrame())
visits = computed.get("visits", pd.DataFrame())
sum_df = computed.get("sum_df", pd.DataFrame())
clust_df = computed.get("clust_df", pd.DataFrame())
avail_df_merge = computed.get("avail_df_merge", pd.DataFrame())
loc_df = computed.get("location_detail", pd.DataFrame()).rename(columns={"City":"city","Cluster":"cluster"})

df = avail_df_merge.copy()
df = df.merge(loc_df[["city","cluster"]], on="city", how="left") if not loc_df.empty else df
df = _ensure_cols(df, {"nearest_end_date": pd.NaT,"joined_dse": 0,"active_dse": 0,"tag": "","availability": "","brand": "","city": "","cluster": "","name": "","latitude": np.nan,"longitude": np.nan})
df["nearest_end_date"] = pd.to_datetime(df["nearest_end_date"], errors="coerce")
df["status_days_to_expire"] = (df["nearest_end_date"] - pd.Timestamp.today().normalize()).dt.days
df["will_expire_30d"] = np.where(df["status_days_to_expire"].between(0,30, inclusive="both"), True, False)
df["tag"] = np.where((pd.to_numeric(df["joined_dse"], errors="coerce").fillna(0)==0) & (pd.to_numeric(df["active_dse"], errors="coerce").fillna(0)==0), "Not Penetrated", df["tag"].replace({np.nan:"Not Active", "": "Not Active"}))
df["availability"] = df["availability"].replace({np.nan:"Potential", "": "Potential"})
df["cluster"] = df["cluster"].astype(str)
df = _coerce_float(df, ["latitude","longitude"])
df = _finite_mask(df, "latitude", "longitude")

if area_pick and "All" not in area_pick:
    df = df[df["cluster"].astype(str).isin(area_pick)]
if city_pick and "All" not in city_pick:
    df = df[df["city"].astype(str).isin(city_pick)]
if brand_sel and "All" not in brand_sel:
    df = df[df["brand"].astype(str).isin(brand_sel)]
if penetrated and "All" not in penetrated:
    df = df[df["tag"].astype(str).isin(penetrated)]
if potential and "All" not in potential:
    df = df[df["availability"].astype(str).isin(potential)]

if df.empty:
    st.info("No dealers match your filters.")
    st.stop()

if bde != "All":
    picks = []
    if not clust_df.empty:
        clusters_for_bde = clust_df["cluster"].dropna().unique().tolist()
        for i in clusters_for_bde:
            col = f"dist_center_{int(i)}"
            if col in df.columns:
                part = df[(pd.to_numeric(df[col], errors="coerce").notna()) & (pd.to_numeric(df[col], errors="coerce") <= radius)].copy()
                if not part.empty:
                    part["cluster_labels"] = int(i)
                    part["sales_name"] = bde
                    picks.append(part)
    df_pick = pd.concat(picks, ignore_index=True) if picks else df.copy()
else:
    df_pick = df.copy()

df_pick["joined_dse"] = pd.to_numeric(df_pick["joined_dse"], errors="coerce").fillna(0)
df_pick["active_dse"] = pd.to_numeric(df_pick["active_dse"], errors="coerce").fillna(0)
df_pick["engagement_bucket"] = np.select(
    [
        (df_pick["active_dse"]>0),
        (df_pick["joined_dse"]>0) & (df_pick["active_dse"]==0),
        (df_pick["joined_dse"]==0) & (df_pick["active_dse"]==0),
    ],
    ["Active","Not Active","Not Penetrated"],
    default="Not Penetrated"
)

clat, clon = _center_of(df_pick, "latitude", "longitude")
color_map = {"Active":[21,255,87,200],"Not Active":[255,171,171,200],"Not Penetrated":[131,201,255,200]}
df_pick["color"] = df_pick["engagement_bucket"].map(color_map).apply(lambda x: x if isinstance(x, list) else [200,200,200,180])

clusters = clust_df.copy()
if not clusters.empty:
    clusters = _ensure_cols(clusters, {"latitude": np.nan, "longitude": np.nan, "sales_name":"", "cluster":0})
    clusters = _coerce_float(clusters, ["latitude","longitude"])
    clusters = _finite_mask(clusters, "latitude", "longitude")
if not clusters.empty:
    visits_cnt = sum_df.groupby(["sales_name","cluster"], as_index=False)["cluster"].count().rename(columns={"cluster":"count_visit"}) if not sum_df.empty else pd.DataFrame(columns=["sales_name","cluster","count_visit"])
    clusters = clusters.merge(visits_cnt, on=["sales_name","cluster"], how="left")
    total_vis = max(float(clusters["count_visit"].fillna(0).sum()), 1.0)
    clusters["size"] = clusters["count_visit"].fillna(0)/total_vis*9000
    clusters["word"] = "Area " + (pd.to_numeric(clusters["cluster"], errors="coerce").fillna(0).astype(int)+1).astype(str) + "\nCount Visit: " + clusters["count_visit"].fillna(0).astype(int).astype(str)
else:
    clusters = pd.DataFrame([{"latitude":clat,"longitude":clon,"size":5000,"word":"Area"}])

st.markdown("<hr style='margin:8px 0 8px 0'/>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)

deck_layers = [
    pdk.Layer("TextLayer", data=clusters, get_position="[longitude,latitude]", get_text="word", get_size=12, get_color=[0,128,0], get_text_anchor=String("middle"), get_alignment_baseline=String("center")),
    pdk.Layer("ScatterplotLayer", data=df_pick, get_position="[longitude,latitude]", get_radius=200, get_fill_color="color", pickable=True, auto_highlight=True),
    pdk.Layer("ScatterplotLayer", data=clusters, get_position="[longitude,latitude]", get_radius="size", get_fill_color=[200,30,0,90]),
]
st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=pdk.ViewState(longitude=clon, latitude=clat, zoom=10, pitch=50), tooltip={"text":"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nActivity: {engagement_bucket}"}, layers=deck_layers))

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Dealers", int(pd.to_numeric(df_pick["id_dealer_outlet"], errors="coerce").nunique()))
with kpi2:
    st.metric("Active Dealers", int((df_pick["active_dse"]>0).sum()))
with kpi3:
    st.metric("Active DSE", int(df_pick["active_dse"].sum()))
with kpi4:
    if not visits.empty and "visit_datetime" in visits.columns:
        wv = visits.dropna(subset=["visit_datetime"]).copy()
        if "matched_dealer_id" in wv.columns:
            wv["week"] = pd.to_datetime(wv["visit_datetime"], errors="coerce").dt.to_period("W").astype(str)
            avg_week = wv.groupby("matched_dealer_id")["week"].nunique().mean()
            st.metric("Avg Weekly Visits", round(float(avg_week if pd.notna(avg_week) else 0),2))
        else:
            st.metric("Avg Weekly Visits", 0)
    else:
        st.metric("Avg Weekly Visits", 0)

st.markdown("<h2 style='font-size:24px;margin:8px 0'>Insights</h2>", unsafe_allow_html=True)
bar_src = df_pick.groupby(["brand","engagement_bucket"], as_index=False)["id_dealer_outlet"].count().rename(columns={"id_dealer_outlet":"Count Dealers","brand":"Brand","engagement_bucket":"Status"})
fig = px.bar(bar_src, x="Brand", y="Count Dealers", color="Status", barmode="group")
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True, key="bar_brand_status")

pie_src = df_pick.groupby(["availability"], as_index=False)["id_dealer_outlet"].count().rename(columns={"id_dealer_outlet":"Total"})
pie = px.pie(pie_src, names="availability", values="Total", hole=0.35)
st.plotly_chart(pie, use_container_width=True, key="pie_avail")

st.markdown("<h2 style='font-size:24px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
show_cols = ["cluster","brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability","will_expire_30d"]
df_table = df_pick[show_cols].rename(columns={"cluster":"Area","brand":"Brand","name":"Dealer Name","city":"City","tag":"Activity","joined_dse":"Total Joined DSE","active_dse":"Total Active DSE","nearest_end_date":"Nearest Package End Date","availability":"Availability","will_expire_30d":"Expire ≤30d"}).drop_duplicates().reset_index(drop=True)
st.dataframe(df_table, use_container_width=True)
