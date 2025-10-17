import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pydeck as pdk
from pydeck.types import String
from data_preprocess import compute_all

st.set_page_config(page_title="Dealer Penetration Dashboard", layout="wide")
st.title("Filter for Recommendation")

with st.spinner("Loading data..."):
    computed = compute_all()

dealers = computed["dealers"]
sum_df = computed["sum_df"]
avail_df_merge = computed["avail_df_merge"]
clust_df = computed["clust_df"]

if dealers.empty or avail_df_merge.empty:
    st.info("Data tidak tersedia.")
    st.stop()

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Nova Handoyo','Muhammad Achlan','Rudy Setya wibowo','Samin Jaya']
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

bde_from_data = sorted(sum_df.get("sales_name", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
if not bde_from_data and "sales_name" in avail_df_merge.columns and avail_df_merge["sales_name"].nunique()==1:
    bde_from_data = ["All"]
elif not bde_from_data:
    bde_from_data = ["All"]

with st.container(border=True):
    name = st.selectbox("BDE Name ", bde_from_data)
    cols1 = st.columns(2)
    with cols1[0]:
        penetrated = st.multiselect("Dealer Activity", ['All','Not Active','Not Penetrated','Active'], default=['All'])
        if "All" in penetrated:
            penetrated = ['Not Active','Not Penetrated','Active']
        radius = st.slider("Choose Radius", 0, 50, 15)
    with cols1[1]:
        potential =  st.multiselect("Dealer Availability", ['All','Potential','Low Generation','Deficit'], default=['All'])
        if "All" in potential:
            potential = ['Potential','Low Generation','Deficit']
        if name in sales_jabo or name=="All":
            jabo = list(avail_df_merge[(avail_df_merge.get("city","").isin(jabodetabek))]['city'].dropna().unique())
            jabo = ["All"] + jabo
            city_pick = st.multiselect("Choose City", jabo, default=["All"])
            if "All" in city_pick:
                city_pick = jabo[1:]
        else:
            regional = list(avail_df_merge[(~avail_df_merge.get("city","").isin(jabodetabek))]['city'].dropna().unique())
            regional = ["All"] + regional
            city_pick = st.multiselect("Choose City", regional, default=["All"])
            if "All" in city_pick:
                city_pick = regional[1:]

    brand_choose = list(
        avail_df_merge[
            ((avail_df_merge.get("sales_name","")==name) | (name=="All")) &
            (avail_df_merge.get("city","").isin(city_pick if city_pick else [])) &
            (avail_df_merge.get("tag","").isin(penetrated)) &
            (avail_df_merge.get("availability","").isin(potential))
        ]["brand"].dropna().unique()
    )
    brand_choose = ["All"] + brand_choose
    brand = st.multiselect("Choose Brand", brand_choose, default=["All"])
    if "All" in brand:
        brand = brand_choose[1:]
    button = st.button("Submit")

if button and name and penetrated and potential and city_pick is not None:
    working = avail_df_merge.copy()
    if name != "All":
        if f"dist_center_0" in working.columns:
            picks = []
            n_clusters = len(sum_df[sum_df.get("sales_name","")==name].get("cluster", pd.Series([], dtype=int)).unique())
            if n_clusters == 0:
                n_clusters = 1
            for i in range(n_clusters):
                col = f"dist_center_{i}"
                if col in working.columns:
                    tp = working[(pd.to_numeric(working[col], errors="coerce") <= float(radius)) & (working.get("sales_name","")==name)].copy()
                    if not tp.empty:
                        tp["cluster_labels"] = i
                        picks.append(tp)
            working = pd.concat(picks, ignore_index=True) if picks else working[working.get("sales_name","")==name].copy()
        else:
            working = working[working.get("sales_name","")==name].copy()
    else:
        working["cluster_labels"] = 0

    drop_cols = [c for c in working.columns if c.startswith("dist_center_")]
    if drop_cols:
        working.drop(columns=drop_cols, inplace=True, errors="ignore")

    working['joined_dse'] = pd.to_numeric(working.get('joined_dse', 0), errors="coerce").fillna(0)
    working['active_dse'] = pd.to_numeric(working.get('active_dse', 0), errors="coerce").fillna(0)
    working['tag'] = np.where((working['joined_dse']==0)&(working['active_dse']==0),"Not Penetrated",working.get('tag','Not Active'))
    working['nearest_end_date'] = pd.to_datetime(working.get('nearest_end_date'), errors="coerce").astype(str)
    working['nearest_end_date'] = np.where(working['nearest_end_date']=='NaT',"No Package Found",working['nearest_end_date'])

    working = working[
        (working.get("city","").isin(city_pick)) &
        (working.get("availability","").isin(potential)) &
        (working.get("tag","").isin(penetrated)) &
        (working.get("brand","").isin(brand if brand else []))
    ].copy()

    dealer_rec = working.copy()
    if dealer_rec.empty:
        st.info("Tidak ada dealer yang memenuhi filter.")
        st.stop()

    if "cluster_labels" not in dealer_rec.columns:
        dealer_rec["cluster_labels"] = 0

    for col in ["delta","latitude","longitude"]:
        if col not in dealer_rec.columns:
            dealer_rec[col] = np.nan
    dealer_rec.sort_values(['cluster_labels','delta','latitude','longitude'],ascending=False,inplace=True)

    if name == "All":
        center_lon = float(pd.to_numeric(dealer_rec["longitude"], errors="coerce").mean())
        center_lat = float(pd.to_numeric(dealer_rec["latitude"], errors="coerce").mean())
        cluster_center = pd.DataFrame([{"latitude":center_lat,"longitude":center_lon,"size":5000,"word":"Area","word_pick":"Area 1","cluster":0}])
    else:
        cluster_center = clust_df[clust_df.get("sales_name","")==name].copy()
        if not cluster_center.empty:
            count_visit_cluster = sum_df[sum_df.get("sales_name","")==name][['cluster','sales_name']].groupby('cluster').count().reset_index().rename(columns={'sales_name':'count_visit'})
            cluster_center = pd.merge(cluster_center, count_visit_cluster, on='cluster', how='left')
            total_vis = max(cluster_center['count_visit'].fillna(0).sum(), 1)
            cluster_center['size'] = cluster_center['count_visit'].fillna(0)/total_vis*9000
            cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
            cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str) + "\nCount Visit: " + cluster_center['count_visit'].fillna(0).astype(int).astype(str)
            cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)
        else:
            center_lon = float(pd.to_numeric(dealer_rec["longitude"], errors="coerce").mean())
            center_lat = float(pd.to_numeric(dealer_rec["latitude"], errors="coerce").mean())
            cluster_center = pd.DataFrame([{"latitude":center_lat,"longitude":center_lon,"size":5000,"word":"Area","word_pick":"Area 1","cluster":0}])

    dealer_rec['area_tag'] = dealer_rec['cluster_labels'].astype(int) + 1
    dealer_rec['area_tag_word'] = "Area " + dealer_rec['area_tag'].astype(str)

    st.title("Penetration Map")
    init_lon = float(pd.to_numeric(cluster_center.get("longitude"), errors="coerce").mean())
    init_lat = float(pd.to_numeric(cluster_center.get("latitude"), errors="coerce").mean())

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(longitude=init_lon, latitude=init_lat, zoom=10, pitch=50),
            tooltip={'text':"Dealer Name: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
            layers=[
                pdk.Layer("TextLayer", data=cluster_center, get_position="[longitude,latitude]", get_text="word", get_size=12, get_color=[0,1000,0], get_angle=0, get_text_anchor=String("middle"), get_alignment_baseline=String("center")),
                pdk.Layer("ScatterplotLayer", data=dealer_rec, get_position="[longitude,latitude]", get_radius=200, get_color="[21, 255, 87, 200]", id="dealer", pickable=True, auto_highlight=True),
                pdk.Layer("ScatterplotLayer", data=cluster_center, get_position="[longitude,latitude]", get_radius="size", get_color="[200, 30, 0, 90]"),
            ]
        ),
        use_container_width=True
    )

    def some_output(area):
        df_output = dealer_rec[dealer_rec['area_tag_word'] == area].copy()
        df_output = df_output[['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']]
        st.markdown(f"### There are {len(df_output)} dealers in the radius {radius} km")
        fig = px.bar(
            df_output[['brand','tag','city']].groupby(['brand','tag']).count().reset_index().rename(columns={'tag':' ','city':'Count Dealers','brand':'Brand'}).sort_values(' '),
            x='Brand', y='Count Dealers', hover_data=['Brand','Count Dealers'],
            color=' ', color_discrete_map={'Not Penetrated':"#83c9ff",'Not Active':'#ffabab','Active':"#ff2b2b"}
        )
        fig.update_layout(legend=dict(orientation='h',yanchor="bottom",y=1.02,xanchor="right",x=1))
        pot_df_output = df_output[['availability','brand','city']].groupby(['availability','brand']).count().reset_index()
        pot_df_output.rename(columns={'availability':'Availability','brand':'Brand','city':'Total Dealers'}, inplace=True)
        fig1 = px.sunburst(pot_df_output, path=['Availability', 'Brand'], values='Total Dealers', color='Availability', color_discrete_map={'Potential':"#83c9ff",'Low Generation':'#ffabab','Deficit':"#ff2b2b"})
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("#### Dealer Penetration"); st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Potential Dealer"); st.plotly_chart(fig1, use_container_width=True)
        st.markdown("### Dealers Details"); st.dataframe(df_output.reset_index(drop=True), use_container_width=True)

    st.title("Dealers Detail")
    tab_labels = sorted(cluster_center.get('word_pick', pd.Series(['Area 1'])).astype(str).unique().tolist())
    for tab, area_label in zip(st.tabs(tab_labels), tab_labels):
        with tab:
            some_output(area_label)
else:
    st.info("Set filter lalu klik Submit.")
