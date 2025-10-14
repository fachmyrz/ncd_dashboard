import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pydeck as pdk
from pydeck.types import String
from data_preprocess import prepare_data

st.set_page_config(page_title="Dealer Penetration", layout="wide")

with st.spinner("Loading and preparing data..."):
    data = prepare_data(pick_date="2024-11-01")
    sum_df = data["sum_df"]
    clust_df = data["clust_df"]
    avail_df_merge = data["avail_df_merge"]

sales_jabo = [
    'A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat',
    'Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Achlan','Samin Jaya'
]
jabodetabek = [
    'Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur',
    'Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang',
    'Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro'
]

st.title("Filter for Recommendation")
with st.container(border=True):
    name = st.selectbox(
        "BDE Name",
        sorted(list(avail_df_merge['sales_name'].dropna().unique())) or
        ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Mikha Dio Arneta','Fahmi Farhan']
    )
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
        if name in sales_jabo:
            jabo = list(avail_df_merge[(avail_df_merge.sales_name == name) & (avail_df_merge.city.isin(jabodetabek))]['city'].dropna().unique())
            jabo = sorted(jabo)
            jabo.insert(0, "All")
            city_pick = st.multiselect("Choose City", jabo, default=["All"])
            if "All" in city_pick:
                city_pick = jabo[1:]
        else:
            regional = list(avail_df_merge[(avail_df_merge.sales_name == name) & (~avail_df_merge.city.isin(jabodetabek))]['city'].dropna().unique())
            regional = sorted(regional)
            regional.insert(0, "All")
            city_pick = st.multiselect("Choose City", regional, default=["All"])
            if "All" in city_pick:
                city_pick = regional[1:]
    brand_choose = list(avail_df_merge[
        (avail_df_merge.sales_name == name) &
        (avail_df_merge.city.isin(city_pick)) &
        (avail_df_merge.tag.isin(penetrated)) &
        (avail_df_merge.availability.isin(potential))
    ]['brand'].dropna().unique())
    brand_choose = sorted(brand_choose)
    brand_choose.insert(0, "All")
    brand = st.multiselect("Choose Brand", brand_choose, default=["All"])
    if "All" in brand:
        brand = brand_choose[1:]
    button = st.button("Submit")

if button and name:
    pick_avail_lst = []
    n_clusters = len(sum_df[sum_df.sales_name == name]['cluster'].unique())
    for i in range(n_clusters):
        temp_pick = avail_df_merge[(avail_df_merge.get(f'dist_center_{i}').notna()) &
                                   (avail_df_merge[f'dist_center_{i}'] <= radius) &
                                   (avail_df_merge.sales_name == name)]
        temp_pick = temp_pick.copy()
        temp_pick['cluster_labels'] = i
        pick_avail_lst.append(temp_pick)
    if pick_avail_lst:
        pick_avail = pd.concat(pick_avail_lst, ignore_index=True)
        dist_cols = [c for c in pick_avail.columns if str(c).startswith('dist_center_')]
        if dist_cols:
            pick_avail.drop(columns=dist_cols, inplace=True, errors='ignore')
        pick_avail['joined_dse'] = pick_avail['joined_dse'].fillna(0)
        pick_avail['active_dse'] = pick_avail['active_dse'].fillna(0)
        pick_avail['tag'] = np.where(
            (pick_avail.joined_dse == 0) & (pick_avail.active_dse == 0),
            "Not Penetrated",
            pick_avail.tag
        )
        pick_avail['nearest_end_date'] = pick_avail['nearest_end_date'].astype(str)
        pick_avail['nearest_end_date'] = np.where(
            pick_avail['nearest_end_date'] == 'NaT', "No Package Found", pick_avail['nearest_end_date']
        )
        if name in sales_jabo:
            pick_avail = pick_avail[pick_avail.cluster == 'Jabodetabek']
        else:
            pick_avail = pick_avail[pick_avail.cluster != 'Jabodetabek']
        dealer_rec = pick_avail[
            (pick_avail.city.isin(city_pick)) &
            (pick_avail.availability.isin(potential)) &
            (pick_avail.tag.isin(penetrated)) &
            (pick_avail.brand.isin(brand))
        ].copy()
        dealer_rec.sort_values(['cluster_labels','delta','latitude','longitude'], ascending=False, inplace=True)
        cluster_center = clust_df[clust_df.sales_name == name].copy()
        count_visit_cluster = (sum_df[sum_df.sales_name == name][['cluster','sales_name']]
                               .groupby('cluster').count().reset_index()
                               .rename(columns={'sales_name':'count_visit'}))
        cluster_center = pd.merge(cluster_center, count_visit_cluster, on='cluster', how='left')
        total_vis = max(cluster_center['count_visit'].sum(), 1)
        cluster_center['size'] = cluster_center['count_visit'] / total_vis * 9000
        cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
        cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str) + "\nCount Visit: " + cluster_center['count_visit'].astype(str)
        cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)
        dealer_rec['area_tag'] = dealer_rec['cluster_labels'].astype(int) + 1
        dealer_rec['area_tag_word'] = "Area " + dealer_rec['area_tag'].astype(str)
        st.title("Penetration Map")
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    longitude=float(cluster_center.longitude.mean()),
                    latitude=float(cluster_center.latitude.mean()),
                    zoom=10, pitch=50
                ),
                tooltip={'text': "Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
                layers=[
                    pdk.Layer(
                        "TextLayer",
                        data=cluster_center,
                        get_position="[longitude,latitude]",
                        get_text="word",
                        get_size=12,
                        get_color=[0, 100, 0],
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
        st.title("Dealers Detail")
        tab_labels = sorted(cluster_center['word_pick'].dropna().unique().tolist())
        tabs = st.tabs(tab_labels if tab_labels else ["No Area"])
        for tab, area_label in zip(tabs, tab_labels):
            with tab:
                df_output = dealer_rec[dealer_rec.area_tag_word == area_label][[
                    'brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability'
                ]].rename(columns={
                    'brand':'Brand',
                    'name':'Name',
                    'city':'City',
                    'tag':'Activity',
                    'joined_dse':'Total Joined DSE',
                    'active_dse':'Total Active DSE',
                    'nearest_end_date':'Nearest Package End Date',
                    'availability':'Availability'
                })
                st.markdown(f"### There are {len(df_output)} dealers in the radius {radius} km")
                if not df_output.empty:
                    bar_src = (df_output[['Brand','Activity','City']]
                               .groupby(['Brand','Activity'])
                               .count().reset_index()
                               .rename(columns={'City':'Count Dealers'}))
                    fig = px.bar(bar_src.sort_values('Activity'),
                                 x='Brand', y='Count Dealers', color='Activity',
                                 hover_data=['Brand','Count Dealers'])
                    st.markdown("#### Dealer Penetration")
                    st.plotly_chart(fig, use_container_width=True)
                    sun_src = (df_output[['Availability','Brand','City']]
                               .groupby(['Availability','Brand']).count().reset_index()
                               .rename(columns={'City':'Total Dealers'}))
                    fig1 = px.sunburst(sun_src, path=['Availability','Brand'], values='Total Dealers', color='Availability')
                    st.markdown("#### Potential Dealer")
                    st.plotly_chart(fig1, use_container_width=True)
                st.markdown("### Dealers Details")
                st.dataframe(df_output.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No dealers found within the selected radius.")
