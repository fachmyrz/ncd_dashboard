import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
from data_preprocess import *
import pydeck as pdk
from pydeck.types import String

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Nova Handoyo','Muhammad Achlan','Rudy Setya wibowo','Samin Jaya']
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

if 'avail_df_merge' not in globals() or avail_df_merge is None:
    avail_df_merge = pd.DataFrame()
if 'sum_df' not in globals() or sum_df is None:
    sum_df = pd.DataFrame()
if 'clust_df' not in globals() or clust_df is None:
    clust_df = pd.DataFrame()

def series(values, length):
    return pd.Series(values, index=range(length)) if length > 0 else pd.Series([], dtype=object)

def get_all_dist_cols(df):
    return [c for c in df.columns if c.startswith('dist_center_')]

st.title("Filter for Recommendation")

name = st.selectbox("BDE Name ", ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Nova Handoyo','Muhammad Achlan','Rudy Setya wibowo','Samin Jaya'])

cols1 = st.columns(2)
with cols1[0]:
    penetrated = st.multiselect("Dealer Activity", ['All','Not Active','Not Penetrated','Active'], default=['All'])
    if "All" in penetrated:
        penetrated = ['Not Active','Not Penetrated','Active']
    radius = st.slider("Choose Radius", 0, 50, 15)
with cols1[1]:
    potential = st.multiselect("Dealer Availability", ['All','Potential','Low Generation','Deficit'], default=['All'])
    if "All" in potential:
        potential = ['Potential','Low Generation','Deficit']

if not avail_df_merge.empty:
    base_for_city = avail_df_merge.copy()
    if 'sales_name' in base_for_city.columns:
        subset = base_for_city[base_for_city['sales_name'] == name]
        if subset.empty:
            subset = base_for_city.copy()
    else:
        subset = base_for_city.copy()
    city_list = subset['city'].dropna().astype(str).unique().tolist()
    if name in sales_jabo:
        jabo_only = [c for c in city_list if c in jabodetabek]
        city_list = jabo_only if len(jabo_only) > 0 else city_list
    city_options = ["All"] + sorted(list(dict.fromkeys(city_list)))
else:
    city_options = ["All"]

city_pick = st.multiselect("Choose City", city_options, default=["All"])
city_filter = None if "All" in city_pick or len(city_pick) == 0 else city_pick

if not avail_df_merge.empty:
    df_brand_base = avail_df_merge.copy()
    if 'sales_name' in df_brand_base.columns:
        df_brand_base = df_brand_base[(df_brand_base['sales_name'] == name) | (~df_brand_base['sales_name'].notna()) | (df_brand_base['sales_name'] == df_brand_base['sales_name'])]
    if city_filter is not None and 'city' in df_brand_base.columns:
        df_brand_base = df_brand_base[df_brand_base['city'].isin(city_filter)]
    if 'tag' in df_brand_base.columns:
        df_brand_base = df_brand_base[df_brand_base['tag'].isin(penetrated)]
    if 'availability' in df_brand_base.columns:
        df_brand_base = df_brand_base[df_brand_base['availability'].isin(potential)]
    brand_opts = df_brand_base['brand'].dropna().astype(str).unique().tolist()
    brand_options = ["All"] + sorted(list(dict.fromkeys(brand_opts))) if len(brand_opts) > 0 else ["All"]
else:
    brand_options = ["All"]

brand = st.multiselect("Choose Brand", brand_options, default=["All"])
brand_filter = None if "All" in brand or len(brand) == 0 else brand

button = st.button("Submit")

if button:
    pick_avail = pd.DataFrame()
    if not avail_df_merge.empty:
        work = avail_df_merge.copy()
        if 'sales_name' in work.columns:
            work = work[work['sales_name'] == name] if name in work['sales_name'].astype(str).unique().tolist() else work
        dist_cols = get_all_dist_cols(work)
        if len(dist_cols) == 0:
            work['dist_center_0'] = np.nan
            dist_cols = ['dist_center_0']
        frames = []
        for i, dc in enumerate(dist_cols):
            cond = (work[dc] <= radius) if pd.api.types.is_numeric_dtype(work[dc]) else series([False], len(work)).astype(bool)
            temp = work[cond].copy()
            if temp.empty and work[dc].isna().all():
                temp = work.copy()
            temp['cluster_labels'] = i
            frames.append(temp)
        pick_avail = pd.concat(frames, ignore_index=True) if len(frames) > 0 else pd.DataFrame()
    if pick_avail.empty:
        pick_avail = avail_df_merge.copy()
        if pick_avail.empty:
            pick_avail = pd.DataFrame(columns=['joined_dse','active_dse','nearest_end_date','tag','cluster','sales_name','city','brand','latitude','longitude'])
        pick_avail['cluster_labels'] = 0

    if 'joined_dse' not in pick_avail.columns:
        pick_avail['joined_dse'] = 0
    if 'active_dse' not in pick_avail.columns:
        pick_avail['active_dse'] = 0
    pick_avail['joined_dse'] = pd.to_numeric(pick_avail['joined_dse'], errors='coerce').fillna(0)
    pick_avail['active_dse'] = pd.to_numeric(pick_avail['active_dse'], errors='coerce').fillna(0)
    if 'tag' not in pick_avail.columns:
        pick_avail['tag'] = 'Not Active'
    pick_avail['tag'] = np.where((pick_avail['joined_dse'] == 0) & (pick_avail['active_dse'] == 0), "Not Penetrated", pick_avail['tag'])
    if 'nearest_end_date' in pick_avail.columns:
        pick_avail['nearest_end_date'] = pick_avail['nearest_end_date'].astype(str).replace(['nan','None','<NA>','NaT'], 'No Package Found')
        pick_avail['nearest_end_date'] = np.where(pick_avail['nearest_end_date'].isin(['nan','None','<NA>','NaT','NaN']), "No Package Found", pick_avail['nearest_end_date'])
    else:
        pick_avail['nearest_end_date'] = "No Package Found"

    if 'cluster' in pick_avail.columns and pick_avail['cluster'].dtype == object:
        if name in sales_jabo:
            pick_avail = pick_avail[pick_avail['cluster'] == 'Jabodetabek'] if 'Jabodetabek' in pick_avail['cluster'].astype(str).unique().tolist() else pick_avail
        else:
            pick_avail = pick_avail[pick_avail['cluster'] != 'Jabodetabek'] if 'Jabodetabek' in pick_avail['cluster'].astype(str).unique().tolist() else pick_avail

    if city_filter is not None and 'city' in pick_avail.columns:
        pick_avail = pick_avail[pick_avail['city'].isin(city_filter)]
    if brand_filter is not None and 'brand' in pick_avail.columns:
        pick_avail = pick_avail[pick_avail['brand'].isin(brand_filter)]
    if 'tag' in pick_avail.columns:
        pick_avail = pick_avail[pick_avail['tag'].isin(penetrated)]
    if 'availability' in pick_avail.columns:
        pick_avail = pick_avail[pick_avail['availability'].isin(potential)]

    dealer_rec = pick_avail.copy()
    if not dealer_rec.empty:
        sort_cols = [c for c in ['cluster_labels','delta','latitude','longitude'] if c in dealer_rec.columns]
        if len(sort_cols) > 0:
            dealer_rec.sort_values(sort_cols, ascending=False, inplace=True)

    cluster_center = clust_df[clust_df.get('sales_name') == name].copy() if not clust_df.empty else pd.DataFrame()
    if not cluster_center.empty and 'cluster' in cluster_center.columns:
        count_visit_cluster = sum_df[sum_df.get('sales_name') == name][['cluster','sales_name']].groupby('cluster').count().reset_index().rename(columns={'sales_name':'count_visit'}) if not sum_df.empty else pd.DataFrame()
        cluster_center = pd.merge(cluster_center, count_visit_cluster, on='cluster', how='left') if not count_visit_cluster.empty else cluster_center
        if 'count_visit' in cluster_center.columns and cluster_center['count_visit'].sum() not in [0, np.nan]:
            cluster_center['size'] = cluster_center['count_visit'] / max(cluster_center['count_visit'].sum(), 1) * 9000
        else:
            cluster_center['size'] = 1000
        cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
        cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str) + "\nCount Visit: " + cluster_center.get('count_visit', series(['0'], len(cluster_center))).astype(str)
        cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)
    else:
        if not dealer_rec.empty and {'latitude','longitude'}.issubset(dealer_rec.columns):
            cluster_center = pd.DataFrame({
                'latitude':[dealer_rec['latitude'].astype(float).mean() if pd.api.types.is_numeric_dtype(dealer_rec['latitude']) else -6.17511],
                'longitude':[dealer_rec['longitude'].astype(float).mean() if pd.api.types.is_numeric_dtype(dealer_rec['longitude']) else 106.827153],
                'size':[1000],
                'word':["Area 1\nCount Visit: 0"],
                'word_pick':["Area 1"],
                'cluster':[0],
                'sales_name':[name]
            })
        else:
            cluster_center = pd.DataFrame({
                'latitude':[-6.175110],
                'longitude':[106.827153],
                'size':[1000],
                'word':["Area 1\nCount Visit: 0"],
                'word_pick':["Area 1"],
                'cluster':[0],
                'sales_name':[name]
            })

    if not dealer_rec.empty and 'cluster_labels' in dealer_rec.columns:
        dealer_rec['area_tag'] = dealer_rec['cluster_labels'].astype(int) + 1
        dealer_rec['area_tag_word'] = "Area " + dealer_rec['area_tag'].astype(str)

    st.title("Penetration Map")

    center_long = cluster_center['longitude'].astype(float).mean()
    center_lat = cluster_center['latitude'].astype(float).mean()

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                longitude=float(center_long),
                latitude=float(center_lat),
                zoom=10,
                pitch=50
            ),
            tooltip={'text': "Dealer Name: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
            layers=[
                pdk.Layer(
                    "TextLayer",
                    data=cluster_center,
                    get_position="[longitude,latitude]",
                    get_text="word",
                    get_size=12,
                    get_color=[0, 1000, 0],
                    get_angle=0,
                    get_text_anchor=String("middle"),
                    get_alignment_baseline=String("center")
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=dealer_rec,
                    get_position="[longitude,latitude]",
                    get_radius=200,
                    get_color="[21, 255, 87, 200]",
                    id="dealer",
                    pickable=True,
                    auto_highlight=True
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=cluster_center,
                    get_position="[longitude,latitude]",
                    get_radius="size",
                    get_color="[200, 30, 0, 90]"
                ),
            ]
        )
    )

    def some_output(area=None):
        if area is None or 'area_tag_word' not in dealer_rec.columns:
            df_output = dealer_rec.copy() if not dealer_rec.empty else pd.DataFrame()
        else:
            df_output = dealer_rec[dealer_rec['area_tag_word'] == area].copy()
        if not df_output.empty:
            keep_cols = [c for c in ['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability'] if c in df_output.columns]
            df_output = df_output[keep_cols]
        else:
            df_output = pd.DataFrame(columns=['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability'])
        st.markdown(f"### There are {len(df_output)} dealers in the radius {radius} km")
        if not df_output.empty and {'brand','tag','city'}.issubset(df_output.columns):
            agg = df_output[['brand','tag','city']].groupby(['brand','tag']).count().reset_index().rename(columns={'tag':' ','city':'Count Dealers','brand':'Brand'}).sort_values(' ')
            fig = px.bar(agg, x='Brand', y='Count Dealers', hover_data=['Brand','Count Dealers'], color=' ')
            pot_df_output = df_output[['availability','brand','city']].groupby(['availability','brand']).count().reset_index() if {'availability','brand','city'}.issubset(df_output.columns) else pd.DataFrame(columns=['Availability','Brand','Total Dealers'])
            if not pot_df_output.empty:
                pot_df_output.rename(columns={'availability':'Availability','brand':'Brand','city':'Total Dealers'}, inplace=True)
                fig1 = px.sunburst(pot_df_output, path=['Availability','Brand'], values='Total Dealers', color='Availability')
            else:
                fig1 = px.sunburst(pd.DataFrame(columns=['Availability','Brand','Total Dealers']), path=['Availability','Brand'], values='Total Dealers')
        else:
            fig = px.bar(pd.DataFrame(columns=['Brand','Count Dealers']), x='Brand', y='Count Dealers')
            fig1 = px.sunburst(pd.DataFrame(columns=['Availability','Brand','Total Dealers']), path=['Availability','Brand'], values='Total Dealers')
        rename_map = {'brand':'Brand','name':'Name','city':'City','tag':'Activity','joined_dse':'Total Joined DSE','active_dse':'Total Active DSE','nearest_end_date':'Nearest Package End Date','availability':'Availability'}
        df_output = df_output.rename(columns={k:v for k,v in rename_map.items() if k in df_output.columns})
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("#### Dealer Penetration")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Potential Dealer")
            st.plotly_chart(fig1, use_container_width=True)
        st.markdown("### Dealers Details")
        st.dataframe(df_output.reset_index(drop=True), use_container_width=True)

    st.title("Dealers Detail")
    tab_labels = cluster_center.get('word_pick', series([], 0)).unique().tolist() if 'word_pick' in cluster_center.columns else []
    tab_labels = [t for t in tab_labels if pd.notna(t)]
    tab_labels = sorted(tab_labels) if len(tab_labels) > 0 else []
    if len(tab_labels) == 0:
        some_output(None)
    else:
        for tab, area_label in zip(st.tabs(tab_labels), tab_labels):
            with tab:
                some_output(area_label)
