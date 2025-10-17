import streamlit as st
st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

st.title("Filter for Recommendation")

with st.container():
    name = st.selectbox("BDE Name ",['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Nova Handoyo','Muhammad Achlan','Rudy Setya wibowo','Samin Jaya'])
    cols1 = st.columns(2)
    with cols1[0]:
        penetrated = st.multiselect("Dealer Activity",['All','Not Active','Not Penetrated','Active'])
        if "All" in penetrated:
            penetrated = ['Not Active','Not Penetrated','Active']
        radius = st.slider("Choose Radius",0,50,15)
    with cols1[1]:
        potential =  st.multiselect("Dealer Availability",['All','Potential','Low Generation','Deficit'])
        if "All" in potential:
            potential = ['Potential','Low Generation','Deficit']
        if name in sales_jabo:
            if not avail_df_merge.empty:
                jabo = list(avail_df_merge[(avail_df_merge.get('sales_name') == name)&(avail_df_merge.get('city').isin(jabodetabek))]['city'].dropna().unique())
            else:
                jabo = []
            jabo = [c for c in jabo]
            jabo.insert(0,"All")
            city_pick = st.multiselect("Choose City",jabo)
            if "All" in city_pick:
                city_pick = jabo
        else:
            if not avail_df_merge.empty:
                regional = list(avail_df_merge[(avail_df_merge.get('sales_name') == name)&(~avail_df_merge.get('city').isin(jabodetabek))]['city'].dropna().unique())
            else:
                regional = []
            regional = [c for c in regional]
            regional.insert(0,"All")
            city_pick = st.multiselect("Choose City", regional)
            if "All" in city_pick:
                city_pick = regional

    if not avail_df_merge.empty:
        brand_choose = list(avail_df_merge[
            (avail_df_merge.get('sales_name') == name) &
            (avail_df_merge.get('city').isin(jabodetabek)) &
            (avail_df_merge.get('city').isin(city_pick if isinstance(city_pick, list) else [])) &
            (avail_df_merge.get('tag').isin(penetrated)) &
            (avail_df_merge.get('availability').isin(potential))
        ]['brand'].dropna().unique())
    else:
        brand_choose = []
    if len(brand_choose) == 0:
        brand_choose = ["All"]
    else:
        brand_choose.insert(0,"All")
    brand = st.multiselect("Choose Brand",brand_choose)
    if "All" in brand:
        brand = brand_choose

    button = st.button("Submit")

if button and name != "" and penetrated != "" and potential != "" and 'city_pick' in locals():
    pick_avail_lst = []
    try:
        clusters_for_name = sum_df[sum_df.get('sales_name') == name].get('cluster', pd.Series(dtype='int')).unique().tolist() if not sum_df.empty else []
    except Exception:
        clusters_for_name = []
    if len(clusters_for_name) == 0:
        clusters_for_name = []
    for i in range(len(clusters_for_name)):
        if not avail_df_merge.empty:
            colname = f'dist_center_{i}'
            cond_dist = avail_df_merge.get(colname, pd.Series([np.nan]*len(avail_df_merge))) <= radius
            cond_sales = avail_df_merge.get('sales_name') == name
            temp_pick = avail_df_merge[cond_dist & cond_sales].copy()
        else:
            temp_pick = pd.DataFrame()
        if temp_pick is not None and not temp_pick.empty:
            temp_pick = temp_pick.copy()
        temp_pick['cluster_labels'] = i
        pick_avail_lst.append(temp_pick)

    pick_avail = pd.concat(pick_avail_lst, ignore_index=True) if len(pick_avail_lst)>0 else pd.DataFrame()

    if pick_avail.empty:
        pick_avail = pd.DataFrame(columns=['joined_dse','active_dse','nearest_end_date','tag','cluster','sales_name','city','brand','latitude','longitude'])

    if 'joined_dse' not in pick_avail.columns:
        pick_avail['joined_dse'] = 0
    if 'active_dse' not in pick_avail.columns:
        pick_avail['active_dse'] = 0
    pick_avail['joined_dse'] = pd.to_numeric(pick_avail.get('joined_dse', 0), errors='coerce').fillna(0)
    pick_avail['active_dse'] = pd.to_numeric(pick_avail.get('active_dse', 0), errors='coerce').fillna(0)
    pick_avail['tag'] = np.where((pick_avail['joined_dse']==0)&(pick_avail['active_dse']==0),"Not Penetrated", pick_avail.get('tag', 'Not Active'))

    if 'nearest_end_date' in pick_avail.columns:
        pick_avail['nearest_end_date'] = pick_avail['nearest_end_date'].astype(str).replace(['nan','None','<NA>','NaT'], 'No Package Found')
        pick_avail['nearest_end_date'] = np.where(pick_avail['nearest_end_date'].isin(['nan','None','<NA>','NaT','NaN']), "No Package Found", pick_avail['nearest_end_date'])
    else:
        pick_avail['nearest_end_date'] = "No Package Found"

    if name in sales_jabo:
        if 'cluster' in pick_avail.columns and pick_avail['cluster'].dtype == object:
            pick_avail = pick_avail[pick_avail['cluster'] == 'Jabodetabek']
    else:
        if 'cluster' in pick_avail.columns and pick_avail['cluster'].dtype == object:
            pick_avail = pick_avail[pick_avail['cluster'] != 'Jabodetabek']

    if penetrated != "" and potential != "" and city_pick != "":
        pick_avail_filter = pick_avail[
            (pick_avail.get('city').isin(city_pick if isinstance(city_pick, list) else [])) &
            (pick_avail.get('availability').isin(potential)) &
            (pick_avail.get('tag').isin(penetrated)) &
            (pick_avail.get('brand').isin(brand))
        ].copy() if not pick_avail.empty else pd.DataFrame()
    else:
        pick_avail_filter = pick_avail.copy()

    dealer_rec = pick_avail_filter.copy() if not pick_avail_filter.empty else pd.DataFrame()

    if not dealer_rec.empty:
        sort_cols = [c for c in ['cluster_labels','delta','latitude','longitude'] if c in dealer_rec.columns]
        if len(sort_cols) > 0:
            dealer_rec.sort_values(sort_cols, ascending=False, inplace=True, errors='ignore')

    cluster_center = clust_df[clust_df.get('sales_name') == name].copy() if not clust_df.empty else pd.DataFrame()
    if not cluster_center.empty and 'cluster' in cluster_center.columns:
        count_visit_cluster = sum_df[sum_df.get('sales_name') == name][['cluster','sales_name']].groupby('cluster').count().reset_index().rename(columns={'sales_name':'count_visit'}) if not sum_df.empty else pd.DataFrame()
        cluster_center = pd.merge(cluster_center, count_visit_cluster, on='cluster', how='left') if not count_visit_cluster.empty else cluster_center
        if 'count_visit' in cluster_center.columns and cluster_center['count_visit'].sum() != 0:
            cluster_center['size'] = cluster_center['count_visit']/cluster_center['count_visit'].sum()*9000
        else:
            cluster_center['size'] = 1000
        cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
        cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str) + "\nCount Visit: " + cluster_center.get('count_visit', pd.Series(['0']*len(cluster_center))).astype(str)
        cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)
    else:
        if not cluster_center.empty:
            cluster_center['size'] = 1000
            cluster_center['area_tag'] = cluster_center.get('cluster', pd.Series([0]*len(cluster_center))).astype(int) + 1
            cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str)
            cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)

    if not dealer_rec.empty and 'cluster_labels' in dealer_rec.columns:
        dealer_rec['area_tag'] = dealer_rec['cluster_labels'].astype(int) + 1
        dealer_rec['area_tag_word'] = "Area " + dealer_rec['area_tag'].astype(str)

    st.title("Penetration Map")

    center_long = cluster_center['longitude'].mean() if not cluster_center.empty and 'longitude' in cluster_center.columns else 106.827153
    center_lat = cluster_center['latitude'].mean() if not cluster_center.empty and 'latitude' in cluster_center.columns else -6.175110

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                longitude=center_long,
                latitude=center_lat,
                zoom=10,
                pitch=50
            ),
            tooltip={'text':"Dealer Name: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
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
        if area is None:
            df_output = dealer_rec.copy() if not dealer_rec.empty else pd.DataFrame()
        else:
            df_output = dealer_rec[dealer_rec.get('area_tag_word') == area].copy() if not dealer_rec.empty else pd.DataFrame()
        if not df_output.empty:
            df_output = df_output[['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']].copy()
        else:
            df_output = pd.DataFrame(columns=['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability'])
        st.markdown(f"### There are {len(df_output)} dealers in the radius {radius} km")
        if not df_output.empty:
            fig = px.bar(df_output[['brand','tag','city']].groupby(['brand','tag']).count().reset_index().rename(columns={'tag':' ','city':'Count Dealers','brand':'Brand'}).sort_values(' '), x='Brand', y='Count Dealers', hover_data=['Brand','Count Dealers'], color=' ', color_discrete_map={'Not Penetrated':"#83c9ff",'Not Active':'#ffabab','Active':"#ff2b2b"})
            fig.update_layout(legend=dict(orientation='h',yanchor="bottom",y=1.02,xanchor="right",x=1))
            pot_df_output = df_output[['availability','brand','city']].groupby(['availability','brand']).count().reset_index()
            pot_df_output.rename(columns={'availability':'Availability','brand':'Brand','city':'Total Dealers'},inplace=True)
            fig1 = px.sunburst(pot_df_output, path=['Availability', 'Brand'], values='Total Dealers', color='Availability',color_discrete_map={'Potential':"#83c9ff",'Low Generation':'#ffabab','Deficit':"#ff2b2b"})
        else:
            fig = px.bar(pd.DataFrame(columns=['Brand','Count Dealers']), x='Brand', y='Count Dealers')
            fig1 = px.sunburst(pd.DataFrame(columns=['Availability','Brand','Total Dealers']), path=['Availability','Brand'], values='Total Dealers')
        df_output = df_output.rename(columns={'brand':'Brand','name':'Name','city':'City','tag':'Activity','joined_dse':'Total Joined DSE','active_dse':'Total Active DSE','nearest_end_date':'Nearest Package End Date','availability':'Availability'})
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("#### Dealer Penetration")
            st.plotly_chart(fig)
        with col2:
            st.markdown("#### Potential Dealer")
            st.plotly_chart(fig1)
        st.markdown("### Dealers Details")
        st.dataframe(df_output.reset_index(drop=True))

    st.title("Dealers Detail")
    tab_labels = cluster_center.get('word_pick', pd.Series([], dtype=object)).unique().tolist() if not cluster_center.empty else []
    tab_labels = [t for t in tab_labels if pd.notna(t)]
    tab_labels.sort()
    if len(tab_labels) == 0:
        some_output(None)
    else:
        for tab,area_label in zip(st.tabs(tab_labels),tab_labels):
            with tab:
                some_output(area_label)
