import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from pydeck.types import String
from PIL import Image
from data_preprocess import compute_all
st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")
computed = compute_all()
sum_df = computed.get("sum_df", pd.DataFrame())
clust_df = computed.get("clust_df", pd.DataFrame())
avail_df_merge = computed.get("avail_df_merge", pd.DataFrame())
df_visit = computed.get("df_visits", pd.DataFrame())
sales_jabo = sorted(["A. Sofyan","Nova Handoyo","Heriyanto","Aditya rifat","Riski Amrullah Zulkarnain","Rudy Setya Wibowo","Muhammad Ahlan","Samin Jaya"])
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']
st.markdown("<h1 style='font-size:40px;margin:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)
with st.container():
    base = df_visit.copy() if not df_visit.empty else pd.DataFrame()
    nik_mask = ~base.get("nik", pd.Series("", dtype=str)).astype(str).str.contains("deleted-", na=False)
    div_mask = ~base.get("divisi", pd.Series("", dtype=str)).astype(str).str.contains("trainer", case=False, na=False)
    try:
        bde_list = sorted(base.loc[nik_mask & div_mask, "employee_name"].dropna().astype(str).unique().tolist())
    except:
        bde_list = sales_jabo
    name = st.selectbox("BDE Name", ["All"] + bde_list, index=0)
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
        if name != "All" and name in sales_jabo:
            cities = avail_df_merge[(avail_df_merge.sales_name==name) & (avail_df_merge.city.isin(jabodetabek))]["city"].dropna().unique().tolist()
        else:
            cities = avail_df_merge[(avail_df_merge.sales_name==name if name!="All" else True) & (~avail_df_merge.city.isin(jabodetabek))]["city"].dropna().unique().tolist()
        cities = sorted(list(set(cities)))
        city_pick = st.multiselect("Choose City", ["All"] + cities, default=["All"])
        if "All" in city_pick:
            city_pick = cities
    brand_choose = avail_df_merge[(avail_df_merge.sales_name==name if name!="All" else True) & (avail_df_merge.city.isin(city_pick) if city_pick else True) & (avail_df_merge.tag.isin(penetrated) if 'tag' in avail_df_merge.columns else True) & (avail_df_merge.availability.isin(potential) if 'availability' in avail_df_merge.columns else True)]
    brand_list = sorted(brand_choose.get("brand", pd.Series("", dtype=str)).dropna().unique().tolist())
    brand = st.multiselect("Choose Brand", ["All"] + brand_list, default=["All"])
    if "All" in brand:
        brand = brand_list
    button = st.button("Submit")
if not button:
    st.info("Set filters and click Submit.")
else:
    if sum_df is None or sum_df.empty:
        st.warning("No summary data found.")
    else:
        pick_avail_lst = []
        unique_clusters = sorted(sum_df.cluster.dropna().unique().tolist()) if "cluster" in sum_df.columns else []
        for i in unique_clusters:
            col = f"dist_center_{i}"
            if col in avail_df_merge.columns:
                temp_pick = avail_df_merge[(avail_df_merge[col].notna()) & (avail_df_merge[col] <= radius) & ((avail_df_merge.sales_name == name) if name!="All" else True)].copy()
                if not temp_pick.empty:
                    temp_pick['cluster_labels'] = i
                    pick_avail_lst.append(temp_pick)
        if not pick_avail_lst:
            st.info(f"No dealers found within {radius} km.")
        else:
            pick_avail = pd.concat(pick_avail_lst, ignore_index=True)
            drop_cols = [c for c in pick_avail.columns if c.startswith("dist_center_")]
            pick_avail = pick_avail.drop(columns=drop_cols, errors='ignore')
            pick_avail['joined_dse'] = pick_avail.get('joined_dse', pd.Series(0)).fillna(0).astype(int)
            pick_avail['active_dse'] = pick_avail.get('active_dse', pd.Series(0)).fillna(0).astype(int)
            pick_avail['tag'] = np.where((pick_avail.joined_dse==0)&(pick_avail.active_dse==0),"Not Penetrated", pick_avail.get('tag', pd.Series("Not Penetrated")))
            pick_avail['nearest_end_date'] = pd.to_datetime(pick_avail.get('nearest_end_date', pd.Series(pd.NaT)), errors="coerce")
            pick_avail['nearest_end_date'] = pick_avail['nearest_end_date'].dt.strftime("%Y-%m-%d")
            if name in sales_jabo:
                pick_avail = pick_avail[pick_avail.cluster == 'Jabodetabek'] if 'cluster' in pick_avail.columns else pick_avail
            else:
                pick_avail = pick_avail[pick_avail.cluster != 'Jabodetabek'] if 'cluster' in pick_avail.columns else pick_avail
            if city_pick:
                pick_avail = pick_avail[pick_avail.city.isin(city_pick)]
            if potential:
                pick_avail = pick_avail[pick_avail.availability.isin(potential)] if 'availability' in pick_avail.columns else pick_avail
            if penetrated:
                pick_avail = pick_avail[pick_avail.tag.isin(penetrated)]
            if brand:
                pick_avail = pick_avail[pick_avail.brand.isin(brand)]
            dealer_rec = pick_avail.copy()
            dealer_rec = dealer_rec.drop_duplicates(subset=["id_dealer_outlet"]) if "id_dealer_outlet" in dealer_rec.columns else dealer_rec
            if dealer_rec.empty:
                st.info("No dealers match the filters.")
            else:
                dealer_rec = dealer_rec.sort_values(['cluster_labels','availability' if 'availability' in dealer_rec.columns else 'brand','latitude','longitude'], ascending=False)
                cluster_center = clust_df[clust_df.sales_name == name] if name!="All" else clust_df.copy()
                count_visit_cluster = sum_df[sum_df.sales_name == name][['cluster','sales_name']].groupby('cluster').count().reset_index().rename(columns={'sales_name':'count_visit'}) if name!="All" else sum_df[['cluster']].groupby('cluster').size().reset_index(name='count_visit')
                if not cluster_center.empty and not count_visit_cluster.empty:
                    cluster_center = cluster_center.merge(count_visit_cluster, on='cluster', how='left')
                    total_visits = max(cluster_center['count_visit'].sum(), 1)
                    cluster_center['size'] = cluster_center['count_visit']/total_visits*9000
                    cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
                    cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str) + "\nCount Visit: " + cluster_center['count_visit'].fillna(0).astype(int).astype(str)
                    cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)
                else:
                    cluster_center = pd.DataFrame([{"longitude": dealer_rec["longitude"].mean(), "latitude": dealer_rec["latitude"].mean()}])
                dealer_rec['area_tag'] = dealer_rec.get('cluster_labels', pd.Series(0)).astype(int) + 1
                dealer_rec['area_tag_word'] = "Area " + dealer_rec['area_tag'].astype(str)
                center_lon = float(cluster_center.longitude.mean()) if 'longitude' in cluster_center.columns else float(dealer_rec.longitude.mean())
                center_lat = float(cluster_center.latitude.mean()) if 'latitude' in cluster_center.columns else float(dealer_rec.latitude.mean())
                def _col_color(row):
                    tag = row.get('tag','Not Penetrated')
                    if tag == "Not Penetrated":
                        return [131,201,255,200]
                    if tag == "Not Active":
                        return [255,171,171,200]
                    if tag == "Active":
                        return [255,43,43,200]
                    return [200,200,200,200]
                dealer_rec['color'] = dealer_rec.apply(_col_color, axis=1).tolist()
                st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
                deck = pdk.Deck(map_style=None, initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50), tooltip={'text':"Dealer Name: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"}, layers=[
                    pdk.Layer("TextLayer", data=cluster_center.to_dict(orient='records') if not cluster_center.empty else [], get_position="[longitude,latitude]", get_text="word", get_size=12, get_color=[0,100,0], get_angle=0, get_text_anchor=String("middle"), get_alignment_baseline=String("center")),
                    pdk.Layer("ScatterplotLayer", data=dealer_rec.to_dict(orient='records'), get_position="[longitude,latitude]", get_radius=200, get_fill_color="color", pickable=True, auto_highlight=True),
                    pdk.Layer("ScatterplotLayer", data=cluster_center.to_dict(orient='records') if not cluster_center.empty else [], get_position="[longitude,latitude]", get_radius="size", get_fill_color=[200,30,0,90])
                ])
                st.pydeck_chart(deck)
                def some_output(area):
                    df_output = dealer_rec[dealer_rec.area_tag_word == area] if 'area_tag_word' in dealer_rec.columns else dealer_rec
                    df_output = df_output[['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']] if set(['brand','name']).issubset(df_output.columns) else df_output
                    st.markdown(f"### There are {len(df_output)} dealers in the radius {radius} km")
                    if not df_output.empty:
                        bar_src = df_output[['brand','tag','city']].groupby(['brand','tag']).size().reset_index(name='Count Dealers')
                        fig = px.bar(bar_src, x='brand', y='Count Dealers', color='tag', labels={'brand':'Brand','tag':'Activity'})
                        fig.update_layout(legend=dict(orientation='h',yanchor="bottom",y=1.02,xanchor="right",x=1))
                        pot_df_output = df_output[['availability','brand','city']].groupby(['availability','brand']).size().reset_index(name='Total Dealers')
                        if not pot_df_output.empty:
                            fig1 = px.sunburst(pot_df_output, path=['availability','brand'], values='Total Dealers', color='availability')
                        else:
                            fig1 = None
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
                    if not df_output.empty:
                        df_shown = df_output.rename(columns={'brand':'Brand','name':'Name','city':'City','tag':'Activity','joined_dse':'Total Joined DSE','active_dse':'Total Active DSE','nearest_end_date':'Nearest Package End Date','availability':'Availability'})
                        df_shown = df_shown.drop_duplicates(subset=['Name']) if 'Name' in df_shown.columns else df_shown
                        st.markdown("### Dealers Details")
                        st.dataframe(df_shown.reset_index(drop=True))
                st.markdown("<h2 style='font-size:20px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
                tab_labels = cluster_center['word_pick'].dropna().unique().tolist() if 'word_pick' in cluster_center.columns else ["All"]
                tab_labels = sorted(tab_labels)
                for tab, area_label in zip(st.tabs(tab_labels), tab_labels):
                    with tab:
                        some_output(area_label)
