import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pydeck as pdk
from pydeck.types import String
import plotly.express as px
from data_preprocess import compute_all
import gspread
import requests
import json

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=":car:", layout="wide")
if "google_creds" not in st.secrets or "sheet_ids" not in st.secrets:
    st.error("Missing google_creds or sheet_ids in Streamlit secrets")
    st.stop()

client = gspread.service_account_from_dict(st.secrets["google_creds"])
sids = st.secrets["sheet_ids"]
def sheet_to_df_by_key(key, tab_name):
    try:
        sh = client.open_by_key(key)
        ws = sh.worksheet(tab_name)
        data = ws.get_all_values()
        if not data or len(data) < 1:
            return pd.DataFrame()
        cols = data[0]
        rows = data[1:]
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        return pd.DataFrame()

need_cluster = sheet_to_df_by_key(sids["need_cluster"], "By Cluster")
location_detail = sheet_to_df_by_key(sids["package_master"], "City Slug")
dealers = sheet_to_df_by_key(sids["dealer_book"], "Dealers")
visits = sheet_to_df_by_key(sids["dealer_book"], "Visits")
running_order = sheet_to_df_by_key(sids["package_master"], "Database")
orders = sheet_to_df_by_key(sids.get("orders_book", ""), "Orders") if sids.get("orders_book","") else pd.DataFrame()
sheets = {"dealers":dealers,"visits":visits,"location":location_detail,"need_cluster":need_cluster,"running_order":running_order,"orders":orders}
computed = compute_all(sheets)
sum_df = computed["sum_df"]
clust_df = computed["clust_df"]
avail_df_merge = computed["avail_df_merge"]
df_visits = computed["df_visits"]
summary = computed["summary"]
data_sum = computed["data_sum"]

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Ahlan','Samin Jaya']
jabodetabek_list = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

st.markdown("<h1 style='font-size:40px;margin:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)
icon_path = "assets/favicon.png"
try:
    icon = Image.open(icon_path)
    st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=icon, layout="wide")
except:
    pass

base = df_visits.copy()
nik_mask = ~base.get("nik", pd.Series("", dtype=str)).astype(str).str.contains("deleted-", na=False)
div_mask = ~base.get("divisi", pd.Series("", dtype=str)).astype(str).str.contains("trainer", na=False)
bde_list = sorted(base.loc[nik_mask & div_mask, "employee_name"].dropna().astype(str).unique().tolist())
bde_list = ["All"] + bde_list
all_areas = avail_df_merge['cluster'].dropna().astype(str).unique().tolist()
area_counts = avail_df_merge.groupby(avail_df_merge['cluster'].astype(str)).size().reset_index(name='cnt')
area_counts = area_counts.sort_values('cnt', ascending=False)
areas_sorted = area_counts['cluster'].tolist()
if not areas_sorted:
    areas_sorted = ["All"]
else:
    areas_sorted = ["All"] + areas_sorted

with st.container():
    cols1 = st.columns([2,1,1])
    with cols1[0]:
        name = st.selectbox("BDE Name", bde_list, index=0)
    with cols1[1]:
        area_pick = st.multiselect("Area", areas_sorted, default=["All"])
    with cols1[2]:
        city_options = sorted(avail_df_merge['city'].dropna().astype(str).unique().tolist())
        city_options = ["All"] + city_options
        city_pick = st.multiselect("City", city_options, default=["All"])
cols2 = st.columns([1,1,1,1])
with cols2[0]:
    penetrated = st.multiselect("Dealer Activity", ['All','Not Active','Not Penetrated','Active'], default=['All'])
with cols2[1]:
    potential = st.multiselect("Dealer Availability", ['All','Potential','Low Generation','Deficit'], default=['All'])
with cols2[2]:
    radius = st.slider("Choose Radius (km)", 0, 50, 15)
with cols2[3]:
    brand_list = sorted(avail_df_merge['brand'].dropna().astype(str).unique().tolist())
    brand_list = ["All"] + brand_list
    brand = st.multiselect("Choose Brand", brand_list, default=["All"])
if "All" in penetrated:
    penetrated = ['Not Active','Not Penetrated','Active']
if "All" in potential:
    potential = ['Potential','Low Generation','Deficit']
if "All" in brand:
    brand = brand_list

submit = st.button("Submit")
if not submit:
    st.info("Set filters and click Submit.")
else:
    pick_avail_lst = []
    for i in range(len(sum_df['cluster'].unique())):
        col = f'dist_center_{i}'
        if col in avail_df_merge.columns:
            temp_pick = avail_df_merge[(avail_df_merge[col].notna()) & (avail_df_merge[col] <= radius)]
            if name != "All":
                temp_pick = temp_pick[temp_pick['sales_name']==name]
            if not temp_pick.empty:
                temp_pick = temp_pick.copy()
                temp_pick['cluster_labels'] = i
                pick_avail_lst.append(temp_pick)
    if not pick_avail_lst:
        st.warning("No dealers found within selected radius and filters.")
    else:
        pick_avail = pd.concat(pick_avail_lst, ignore_index=True)
        drop_cols = [c for c in pick_avail.columns if c.startswith('dist_center_')]
        pick_avail = pick_avail.drop(columns=drop_cols, errors='ignore')
        pick_avail['joined_dse'] = pick_avail.get('joined_dse',0).fillna(0).astype(int)
        pick_avail['active_dse'] = pick_avail.get('active_dse',0).fillna(0).astype(int)
        pick_avail['tag'] = np.where((pick_avail.joined_dse==0)&(pick_avail.active_dse==0),"Not Penetrated", pick_avail.get('tag','Not Active'))
        pick_avail['nearest_end_date'] = pd.to_datetime(pick_avail.get('nearest_end_date', pd.NaT), errors='coerce')
        if name in sales_jabo:
            pick_avail = pick_avail[pick_avail['cluster'].astype(str) == 'Jabodetabek']
        else:
            pick_avail = pick_avail[pick_avail['cluster'].astype(str) != 'Jabodetabek']
        if area_pick and "All" not in area_pick:
            pick_avail = pick_avail[pick_avail['cluster'].astype(str).isin(area_pick)]
        if city_pick and "All" not in city_pick:
            pick_avail = pick_avail[pick_avail['city'].astype(str).isin(city_pick)]
        pick_avail = pick_avail[pick_avail['availability'].isin(potential)]
        pick_avail = pick_avail[pick_avail['tag'].isin(penetrated)]
        pick_avail = pick_avail[pick_avail['brand'].astype(str).isin(brand)]
        dealer_rec = pick_avail.copy()
        dealer_rec = dealer_rec.sort_values(['cluster_labels','delta','latitude','longitude'], ascending=[True,False,False,False])
        if dealer_rec.empty:
            st.warning("No dealers match the filters after applying all conditions.")
        else:
            dealer_rec = dealer_rec.reset_index(drop=True)
            cluster_center = clust_df[clust_df['sales_name'] == (name if name!="All" else clust_df['sales_name'].mode().iat[0] if not clust_df.empty else "")]
            if cluster_center.empty:
                center_lon = dealer_rec['longitude'].mean()
                center_lat = dealer_rec['latitude'].mean()
            else:
                center_lon = cluster_center['longitude'].mean()
                center_lat = cluster_center['latitude'].mean()
            dealer_rec['color_bucket'] = dealer_rec['tag'].map({'Not Penetrated':[131,197,255,200],'Not Active':[255,186,186,200],'Active':[255,43,43,200]}).tolist() if not dealer_rec.empty else []
            st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
            st.pydeck_chart(
                pdk.Deck(
                    map_style=None,
                    initial_view_state=pdk.ViewState(
                        longitude=float(center_lon),
                        latitude=float(center_lat),
                        zoom=10,
                        pitch=40
                    ),
                    tooltip={'text':"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
                    layers=[
                        pdk.Layer(
                            "TextLayer",
                            data=cluster_center,
                            get_position="[longitude,latitude]",
                            get_text="cluster",
                            get_size=16,
                            get_color=[0,0,0],
                            get_angle=0,
                            get_text_anchor=String("middle"),
                            get_alignment_baseline=String("center")
                        ),
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=dealer_rec,
                            get_position=["longitude","latitude"],
                            get_radius=200,
                            get_fill_color="color",
                            pickable=True,
                            auto_highlight=True,
                            get_color="[21,255,87,200]"
                        ),
                    ]
                )
            )
            def some_output(area):
                df_output = dealer_rec[dealer_rec['cluster'].astype(str).str.contains(area.split()[-1]) if isinstance(area,str) and area!='All' else dealer_rec.index==dealer_rec.index]
                df_output = dealer_rec.copy() if area=='All' else dealer_rec[dealer_rec['cluster'].astype(str).str.contains(area)]
                df_out = df_output[['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']].copy()
                st.markdown(f"### There are {len(df_out)} dealers in the radius {radius} km")
                if not df_out.empty:
                    bar_src = df_out[['brand','tag','city']].groupby(['brand','tag']).count().reset_index().rename(columns={'tag':' ','city':'Count Dealers','brand':'Brand'}).sort_values(' ')
                    fig = px.bar(bar_src, x='Brand', y='Count Dealers', hover_data=['Brand','Count Dealers'], color=' ', color_discrete_map={'Not Penetrated':"#83c9ff",'Not Active':'#ffabab','Active':"#ff2b2b"})
                    fig.update_layout(legend=dict(orientation='h',yanchor="bottom",y=1.02,xanchor="right",x=1))
                    pot_df_output = df_out[['availability','brand','city']].groupby(['availability','brand']).count().reset_index().rename(columns={'availability':'Availability','brand':'Brand','city':'Total Dealers'})
                    fig1 = px.sunburst(pot_df_output, path=['Availability','Brand'], values='Total Dealers', color='Availability', color_discrete_map={'Potential':"#83c9ff",'Low Generation':'#ffabab','Deficit':"#ff2b2b"}) if not pot_df_output.empty else None
                else:
                    fig = None
                    fig1 = None
                col1, col2 = st.columns([2,1])
                with col1:
                    if fig is not None:
                        st.markdown("#### Dealer Penetration")
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if fig1 is not None:
                        st.markdown("#### Potential Dealer")
                        st.plotly_chart(fig1, use_container_width=True)
                if not df_out.empty:
                    df_shown = df_out.rename(columns={'brand':'Brand','name':'Dealer Name','city':'City','tag':'Activity','joined_dse':'Total Joined DSE','active_dse':'Total Active DSE','nearest_end_date':'Nearest Package End Date','availability':'Availability'})
                    df_shown = df_shown.drop_duplicates(subset=['Dealer Name']).reset_index(drop=True)
                    st.markdown("### Dealers Details")
                    st.dataframe(df_shown, use_container_width=True)
            st.title("Dealers Detail")
            tab_labels = sorted(dealer_rec['cluster'].dropna().astype(str).unique().tolist())
            if not tab_labels:
                tab_labels = ["All"]
            for tab, area_label in zip(st.tabs(["All"]+tab_labels), ["All"]+tab_labels):
                with tab:
                    some_output(area_label)
