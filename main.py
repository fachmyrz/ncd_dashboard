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
            jabo = list(avail_df_merge[(avail_df_merge.sales_name == name)&(avail_df_merge.city.isin(jabodetabek))]['city'].unique()) if not avail_df_merge.empty else []
            jabo.insert(0,"All")
            city_pick = st.multiselect("Choose City",jabo)
            if "All" in city_pick:
                city_pick = jabo
        else:
            regional = list(avail_df_merge[(avail_df_merge.sales_name == name)&(~avail_df_merge.city.isin(jabodetabek))]['city'].unique()) if not avail_df_merge.empty else []
            regional.insert(0,"All")
            city_pick = st.multiselect("Choose City", regional)
            if "All" in city_pick:
                city_pick = regional

    brand_choose = list(avail_df_merge[(avail_df_merge.sales_name == name)&(avail_df_merge.city.isin(jabodetabek))&(avail_df_merge.city.isin(city_pick))&(avail_df_merge.tag.isin(penetrated))&(avail_df_merge.availability.isin(potential))]['brand'].unique()) if not avail_df_merge.empty else []
    if len(brand_choose) == 0:
        brand_choose = ["All"]
    else:
        brand_choose.insert(0,"All")
    brand = st.multiselect("Choose Brand",brand_choose)
    if "All" in brand:
        brand = brand_choose

    button = st.button("Submit")

if button and name != "" and penetrated != "" and potential != "" and city_pick != "":
    pick_avail_lst = []
    clusters_for_name = sum_df[sum_df.sales_name == name].cluster.unique() if not sum_df.empty else []
    for i in range(len(clusters_for_name)):
        temp_pick = avail_df_merge[(avail_df_merge.get(f'dist_center_{i}', pd.Series([np.nan]*len(avail_df_merge))) <= radius)&(avail_df_merge.sales_name == name)] if not avail_df_merge.empty else pd.DataFrame()
        temp_pick = temp_pick.copy()
        temp_pick['cluster_labels'] = i
        pick_avail_lst.append(temp_pick)
    pick_avail = pd.concat(pick_avail_lst, ignore_index=True) if len(pick_avail_lst)>0 else pd.DataFrame()
    if not pick_avail.empty and any(f'dist_center_{i}' in pick_avail.columns for i in range(len(sum_df.cluster.unique()) if not sum_df.empty else 0)):
        cols_to_drop = [c for c in pick_avail.columns if c.startswith('dist_center_')]
        pick_avail.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    pick_avail['joined_dse'] = pick_avail.get('joined_dse', pd.Series()).fillna(0)
    pick_avail['active_dse'] = pick_avail.get('active_dse', pd.Series()).fillna(0)
    pick_avail['tag'] = np.where((pick_avail.get('joined_dse',0)==0)&(pick_avail.get('active_dse',0)==0),"Not Penetrated",pick_avail.get('tag'))
    pick_avail['nearest_end_date'] = pick_avail.get('nearest_end_date').astype(str)
    pick_avail['nearest_end_date'] = np.where(pick_avail['nearest_end_date'] == 'NaT',"No Package Found",pick_avail['nearest_end_date'])
    if name in sales_jabo:
        pick_avail = pick_avail[pick_avail.cluster == 'Jabodetabek'] if 'cluster' in pick_avail.columns else pick_avail
    else:
        pick_avail = pick_avail[pick_avail.cluster != 'Jabodetabek'] if 'cluster' in pick_avail.columns else pick_avail
    if penetrated != "" and potential != "" and city_pick != "":
        pick_avail_filter = pick_avail[(pick_avail.city.isin(city_pick))&(pick_avail.availability.isin(potential))&(pick_avail.tag.isin(penetrated))&(pick_avail.brand.isin(brand))] if not pick_avail.empty else pd.DataFrame()
    else:
        pick_avail_filter = pick_avail
    dealer_rec = pick_avail_filter.copy() if not pick_avail_filter.empty else pd.DataFrame()
    if not dealer_rec.empty:
        dealer_rec.sort_values(['cluster_labels','delta','latitude','longitude'],ascending=False,inplace=True, errors='ignore')
    cluster_center = clust_df[clust_df.sales_name == name] if not clust_df.empty else pd.DataFrame()
    count_visit_cluster = sum_df[sum_df.sales_name == name][['cluster','sales_name']].groupby('cluster').count().reset_index().rename(columns={'sales_name':'count_visit'}) if not sum_df.empty else pd.DataFrame()
    if not cluster_center.empty and not count_visit_cluster.empty:
        cluster_center = pd.merge(cluster_center,count_visit_cluster,on='cluster',how='left')
        cluster_center['size'] = cluster_center['count_visit']/cluster_center['count_visit'].sum()*9000 if cluster_center['count_visit'].sum() != 0 else 1000
        cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
        cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str) + "\nCount Visit: " + cluster_center['count_visit'].astype(str)
        cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)
    else:
        cluster_center = cluster_center.copy()
        if not cluster_center.empty:
            cluster_center['size'] = 1000
            cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
            cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str)
            cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)
    if not dealer_rec.empty and 'cluster_labels' in dealer_rec.columns:
        dealer_rec['area_tag'] = dealer_rec['cluster_labels'].astype(int) + 1
        dealer_rec['area_tag_word'] = "Area " + dealer_rec['area_tag'].astype(str)
    st.title("Penetration Map")
    center_long = cluster_center['longitude'].mean() if not cluster_center.empty else 106.827153
    center_lat = cluster_center['latitude'].mean() if not cluster_center.empty else -6.175110
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
    def some_output(area):
        df_output = dealer_rec[dealer_rec.area_tag_word == area] if not dealer_rec.empty else pd.DataFrame()
        df_output = df_output[['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']].copy() if not df_output.empty else pd.DataFrame()
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
    tab_labels = cluster_center['word_pick'].unique().tolist() if not cluster_center.empty else []
    tab_labels.sort()
    for tab,area_label in zip(st.tabs(tab_labels),tab_labels):
        with tab:
            some_output(area_label)
