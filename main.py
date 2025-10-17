import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pydeck as pdk
from pydeck.types import String
import plotly.express as px
from data_preprocess import compute_all, haversine_vec

# minimal page config
try:
    st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")
except Exception:
    st.set_page_config(page_title="Dealer Penetration Dashboard", layout="wide")

computed = compute_all()
sum_df = computed.get("sum_df", pd.DataFrame())
clust_df = computed.get("clust_df", pd.DataFrame())
avail_df_merge = computed.get("avail_df_merge", pd.DataFrame())
df_visits = computed.get("df_visits", pd.DataFrame())
revenue_monthly = computed.get("revenue_monthly", pd.DataFrame())

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Achlan','Samin Jaya']
jabodetabek_cities = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

st.markdown("<h1 style='font-size:40px;margin:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)

with st.container():
    base = df_visits.copy() if df_visits is not None else pd.DataFrame()
    nik_mask = ~base.get("nik", pd.Series("", dtype=str)).astype(str).str.contains("deleted-", case=False, na=False)
    div_mask = ~base.get("divisi", pd.Series("", dtype=str)).astype(str).str.contains("Trainer", case=False, na=False)
    bde_list = sorted(base.loc[nik_mask & div_mask, "employee_name"].dropna().astype(str).unique().tolist()) if not base.empty else []
    name = st.selectbox("BDE Name", ["All"] + bde_list, index=0)
    cols1 = st.columns(2)
    with cols1[0]:
        penetrated = st.multiselect("Dealer Activity", ['All','Not Active','Not Penetrated','Active'], default=['All'])
        if "All" in penetrated:
            penetrated = ['Not Active','Not Penetrated','Active']
        radius = st.slider("Choose Radius (km)", 0, 50, 15)
    with cols1[1]:
        potential = st.multiselect("Dealer Availability", ['All','Potential','Low Generation','Deficit'], default=['All'])
        if "All" in potential:
            potential = ['Potential','Low Generation','Deficit']
    all_areas = []
    if avail_df_merge is not None and not avail_df_merge.empty and "cluster" in avail_df_merge.columns:
        tmp = avail_df_merge.groupby("cluster").id_dealer_outlet.nunique().reset_index().rename(columns={"id_dealer_outlet":"count"})
        tmp = tmp.sort_values("count", ascending=False)
        all_areas = tmp["cluster"].astype(str).tolist()
    default_area = ["Jabodetabek"] if "Jabodetabek" in all_areas else (all_areas[:1] if all_areas else [])
    area = st.multiselect("Area (cluster)", ["All"] + all_areas, default=["All"] + default_area)
    if "All" in area:
        area = all_areas
    city_choices = []
    if name and name != "All" and (avail_df_merge is not None and not avail_df_merge.empty):
        city_choices = avail_df_merge[avail_df_merge.get("sales_name","") == name]["city"].dropna().unique().tolist()
    else:
        city_choices = avail_df_merge["city"].dropna().unique().tolist() if (avail_df_merge is not None and not avail_df_merge.empty) else []
    if not city_choices:
        city_choices = []
    city_pick = st.multiselect("Choose City", ["All"] + sorted(city_choices), default=["All"] + sorted(city_choices) if city_choices else [])
    if "All" in city_pick:
        city_pick = city_choices
    brand_choices = avail_df_merge["brand"].dropna().unique().tolist() if (avail_df_merge is not None and not avail_df_merge.empty) else []
    brand = st.multiselect("Choose Brand", ["All"] + sorted(brand_choices), default=["All"] + sorted(brand_choices) if brand_choices else [])
    if "All" in brand:
        brand = brand_choices
    button = st.button("Submit")

def compute_center(name, visits_df, avail_df):
    if name == "All" or not name:
        if avail_df is not None and "cluster" in avail_df.columns and "Jabodetabek" in avail_df["cluster"].unique():
            c = avail_df[avail_df["cluster"] == "Jabodetabek"]
            if not c.empty and "longitude" in c.columns and "latitude" in c.columns:
                return float(c["longitude"].mean()), float(c["latitude"].mean())
        if not visits_df.empty and "long" in visits_df.columns and "lat" in visits_df.columns:
            return float(visits_df["long"].mean()), float(visits_df["lat"].mean())
        return 106.84513, -6.21462
    sel = visits_df[visits_df["employee_name"]==name] if not visits_df.empty else pd.DataFrame()
    if sel.empty:
        if not visits_df.empty and "long" in visits_df.columns and "lat" in visits_df.columns:
            return float(visits_df["long"].mean()), float(visits_df["lat"].mean())
        return 106.84513, -6.21462
    return float(sel["long"].mean()), float(sel["lat"].mean())

if button:
    lon_center, lat_center = compute_center(name, df_visits, avail_df_merge)
    if avail_df_merge is None or avail_df_merge.empty:
        st.info("No dealer data available")
    else:
        dealers = avail_df_merge.copy()
        if brand:
            dealers = dealers[dealers["brand"].isin(brand)]
        if city_pick:
            dealers = dealers[dealers["city"].isin(city_pick)]
        if area:
            dealers = dealers[dealers["cluster"].astype(str).isin([str(x) for x in area])]
        if "latitude" in dealers.columns and "longitude" in dealers.columns:
            lat_arr = dealers["latitude"].fillna(0).to_numpy(dtype=float)
            lon_arr = dealers["longitude"].fillna(0).to_numpy(dtype=float)
            center_lat_arr = np.full_like(lat_arr, float(lat_center))
            center_lon_arr = np.full_like(lon_arr, float(lon_center))
            dealers["dist_center"] = haversine_vec(center_lat_arr, center_lon_arr, lat_arr, lon_arr)
        else:
            dealers["dist_center"] = np.nan
        dealers = dealers[(dealers["dist_center"] <= radius) | (dealers["dist_center"].isna())]
        dealers["joined_dse"] = dealers.get("joined_dse", 0).fillna(0).astype(int)
        dealers["active_dse"] = dealers.get("active_dse", 0).fillna(0).astype(int)
        dealers["tag"] = np.where((dealers["joined_dse"]==0)&(dealers["active_dse"]==0),"Not Penetrated", dealers.get("tag","Not Penetrated"))
        if name in sales_jabo:
            dealers = dealers[dealers["cluster"].astype(str) == "Jabodetabek"]
        else:
            dealers = dealers[dealers["cluster"].astype(str) != "Jabodetabek"]
        if potential:
            dealers = dealers[dealers.get("availability","").isin(potential)]
        if penetrated:
            dealers = dealers[dealers.get("tag","").isin(penetrated)]
        if dealers.empty:
            st.info(f"No dealers found matching filters within {radius} km")
        else:
            dealers = dealers.sort_values(["dist_center","id_dealer_outlet"], ascending=True)
            centers = clust_df.copy()
            st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
            dealers["color"] = dealers.apply(lambda r: [21,255,87,200] if r.get("tag","")=="Active" else ([131,201,255,200] if r.get("tag","")=="Not Penetrated" else [255,170,170,200]), axis=1)
            # ensure color lists
            dealers["color"] = dealers["color"].apply(lambda x: x if isinstance(x,(list,tuple,np.ndarray)) else [200,200,200,150])
            text_layer = pdk.Layer(
                "TextLayer",
                data=centers if not centers.empty else pd.DataFrame(),
                get_position="[longitude,latitude]",
                get_text="cluster",
                get_size=12,
                get_color=[0,0,0],
                get_angle=0,
                get_text_anchor=String("middle"),
                get_alignment_baseline=String("center")
            )
            scatter = pdk.Layer(
                "ScatterplotLayer",
                data=dealers,
                get_position="[longitude,latitude]",
                get_radius=200,
                get_color="color",
                pickable=True,
                auto_highlight=True
            )
            center_layer = pdk.Layer(
                "ScatterplotLayer",
                data=centers if not centers.empty else pd.DataFrame(),
                get_position="[longitude,latitude]",
                get_radius=centers["count_dealers"].fillna(100).to_list() if (not centers.empty and "count_dealers" in centers.columns) else 100,
                get_color=[200,30,0,90]
            )
            view = pdk.ViewState(longitude=lon_center, latitude=lat_center, zoom=10, pitch=40)
            deck = pdk.Deck(map_style=None, initial_view_state=view, layers=[text_layer, scatter, center_layer], tooltip={'text':"Dealer Name: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"})
            st.pydeck_chart(deck)
            def area_output(df):
                df_out = df[['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']].drop_duplicates(subset=['name'])
                st.markdown(f"### There are {len(df_out)} dealers in selection")
                if not df_out.empty:
                    bar_src = df_out.groupby(['brand','tag']).size().reset_index(name='Count Dealers')
                    fig = px.bar(bar_src, x='brand', y='Count Dealers', color='tag', labels={'brand':"Brand"}, color_discrete_map={'Not Penetrated':"#83c9ff",'Not Active':'#ffabab','Active':"#83ff8a"})
                    st.plotly_chart(fig, use_container_width=True)
                    pot_df_output = df_out.groupby(['availability','brand']).size().reset_index(name='Total Dealers')
                    if not pot_df_output.empty:
                        fig1 = px.sunburst(pot_df_output, path=['availability','brand'], values='Total Dealers', color='availability', color_discrete_map={'Potential':"#83c9ff",'Low Generation':'#ffabab','Deficit':"#ff2b2b"})
                        st.plotly_chart(fig1, use_container_width=True)
                df_shown = df_out.rename(columns={'brand':'Brand','name':'Dealer Name','city':'City','tag':'Activity','joined_dse':'Total Joined DSE','active_dse':'Total Active DSE','nearest_end_date':'Nearest Package End Date','availability':'Availability'})
                st.dataframe(df_shown.reset_index(drop=True))
            st.markdown("<h2 style='font-size:20px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
            area_output(dealers)

