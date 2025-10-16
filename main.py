import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocess import *
import streamlit as st
from PIL import Image
import pydeck as pdk
from pydeck.types import String

icon = Image.open("assets/favicon.png")
st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=icon, layout="wide")
st.markdown("""
<style>
h1, .stMarkdown h1 {font-size: 28px !important;}
h2, .stMarkdown h2 {font-size: 24px !important;}
h3, .stMarkdown h3 {font-size: 20px !important;}
</style>
""", unsafe_allow_html=True)

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Ahlan','Samin Jaya']
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

st.markdown("## Filter for Recommendation")

with st.container(border=True):
    base = df_visit.copy()
    if "Nomor Induk Karyawan" not in base.columns:
        base["Nomor Induk Karyawan"] = ""
    if "Divisi" not in base.columns:
        base["Divisi"] = ""
    base["Nomor Induk Karyawan"] = base["Nomor Induk Karyawan"].astype(str)
    base["Divisi"] = base["Divisi"].astype(str)
    nik_mask = ~base["Nomor Induk Karyawan"].str.contains("^deleted-", case=False, na=False)
    div_mask = ~base["Divisi"].str.contains("Trainer", case=False, na=False)
    bde_list = sorted(base.loc[nik_mask & div_mask, "employee_name"].dropna().astype(str).unique().tolist())
    bde = st.selectbox("BDE Name", ["All"] + bde_list, index=0)

    all_areas = sorted(avail_df_merge["cluster"].dropna().astype(str).unique().tolist())
    area_pick = st.multiselect("Area", ["All"] + all_areas, default=["All"])
    if "All" in area_pick:
        area_sel = all_areas
    else:
        area_sel = area_pick

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
        if bde != "All" and bde in sales_jabo:
            jabo = list(avail_df_merge[(avail_df_merge.sales_name == bde)&(avail_df_merge.city.isin(jabodetabek))&(avail_df_merge["cluster"].astype(str).isin(area_sel))]['city'].dropna().unique())
            jabo = sorted(jabo)
            city_pick = st.multiselect("Choose City", ["All"] + jabo, default=["All"])
            if "All" in city_pick:
                city_pick = jabo
        elif bde != "All":
            regional = list(avail_df_merge[(avail_df_merge.sales_name == bde)&(~avail_df_merge.city.isin(jabodetabek))&(avail_df_merge["cluster"].astype(str).isin(area_sel))]['city'].dropna().unique())
            regional = sorted(regional)
            city_pick = st.multiselect("Choose City", ["All"] + regional, default=["All"])
            if "All" in city_pick:
                city_pick = regional
        else:
            all_cities = sorted(avail_df_merge[avail_df_merge["cluster"].astype(str).isin(area_sel)]["city"].dropna().unique().tolist())
            city_pick = st.multiselect("Choose City", ["All"] + all_cities, default=["All"])
            if "All" in city_pick:
                city_pick = all_cities

    brand_choose = list(
        avail_df_merge[
            avail_df_merge["cluster"].astype(str).isin(area_sel) &
            (avail_df_merge.city.isin(city_pick if len(city_pick)>0 else avail_df_merge["city"])) &
            (avail_df_merge.tag.isin(penetrated)) &
            (avail_df_merge.availability.isin(potential))
        ]['brand'].dropna().unique()
    )
    brand_choose = sorted(brand_choose)
    brand = st.multiselect("Choose Brand", ["All"] + brand_choose, default=["All"])
    if "All" in brand:
        brand = brand_choose

    button = st.button("Submit")

if button:
    if bde == "All":
        names = sum_df["sales_name"].dropna().unique().tolist()
    else:
        names = [bde]
    pick_avail_lst = []
    for nm in names:
        clus = sum_df[sum_df.sales_name == nm]["cluster"].dropna().unique().tolist()
        for i in clus:
            col = f"dist_center_{i}"
            if col in avail_df_merge.columns:
                temp_pick = avail_df_merge[(avail_df_merge[col].notna())&(avail_df_merge[col] <= radius)&(avail_df_merge.sales_name == nm)].copy()
                if not temp_pick.empty:
                    temp_pick["cluster_labels"] = i
                    pick_avail_lst.append(temp_pick)
    if not pick_avail_lst:
        st.info(f"No dealers found within {radius} km.")
    else:
        pick_avail = pd.concat(pick_avail_lst, ignore_index=True)
        drop_cols = [c for c in pick_avail.columns if str(c).startswith("dist_center_")]
        if drop_cols:
            pick_avail.drop(columns=drop_cols, inplace=True, errors='ignore')
        pick_avail["joined_dse"] = pick_avail["joined_dse"].fillna(0)
        pick_avail["active_dse"] = pick_avail["active_dse"].fillna(0)
        pick_avail["tag"] = np.where((pick_avail.joined_dse==0)&(pick_avail.active_dse==0),"Not Penetrated",pick_avail.tag)
        pick_avail["nearest_end_date"] = pick_avail["nearest_end_date"].astype(str)
        pick_avail["nearest_end_date"] = np.where(pick_avail["nearest_end_date"] == 'NaT',"No Package Found",pick_avail["nearest_end_date"])
        if bde != "All" and bde in sales_jabo:
            pick_avail = pick_avail[pick_avail.cluster == 'Jabodetabek']
        elif bde != "All":
            pick_avail = pick_avail[pick_avail.cluster != 'Jabodetabek']
        pick_avail = pick_avail[pick_avail["cluster"].astype(str).isin(area_sel)]
        if len(city_pick)>0:
            pick_avail = pick_avail[pick_avail.city.isin(city_pick)]
        pick_avail = pick_avail[pick_avail.availability.isin(potential)]
        pick_avail = pick_avail[pick_avail.tag.isin(penetrated)]
        pick_avail = pick_avail[pick_avail.brand.isin(brand) if len(brand)>0 else True]
        dealer_rec = pick_avail.copy()

        if dealer_rec.empty:
            st.info(f"No dealers match the filters within {radius} km.")
        else:
            dealer_rec.sort_values(['cluster_labels','delta','latitude','longitude'],ascending=False,inplace=True)
            if bde == "All":
                cluster_center = clust_df[clust_df["sales_name"].isin(names)].copy()
                count_visit_cluster = sum_df[sum_df["sales_name"].isin(names)][["cluster","sales_name"]].groupby('cluster').count().reset_index().rename(columns={'sales_name':'count_visit'})
            else:
                cluster_center = clust_df[clust_df.sales_name == bde].copy()
                count_visit_cluster = sum_df[sum_df.sales_name == bde][['cluster','sales_name']].groupby('cluster').count().reset_index().rename(columns={'sales_name':'count_visit'})
            cluster_center = pd.merge(cluster_center,count_visit_cluster,on='cluster',how='left')
            total_visits = max(cluster_center['count_visit'].fillna(0).sum(), 1)
            cluster_center['size'] = cluster_center['count_visit'].fillna(0)/total_visits*9000
            cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
            cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str) + "\nCount Visit: " + cluster_center['count_visit'].fillna(0).astype(int).astype(str)
            cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)
            dealer_rec['area_tag'] = dealer_rec['cluster_labels'].astype(int) + 1
            dealer_rec['area_tag_word'] = "Area " + dealer_rec['area_tag'].astype(str)

            last_56 = pd.Timestamp.today().date() - pd.Timedelta(days=56)
            v = df_visit.copy()
            v = v[pd.to_datetime(v["date"], errors="coerce")>=pd.to_datetime(last_56)]
            wk = v.groupby("client_name").size().reindex(dealer_rec["name"].unique(), fill_value=0)/8.0
            m = dealer_rec.merge(wk.rename("weekly_visits"), left_on="name", right_index=True, how="left")
            kpi_dealers = len(m)
            kpi_active_dealers = (m["active_dse"].fillna(0)>0).sum()
            kpi_active_dse = int(m["active_dse"].fillna(0).sum())
            kpi_avg_weekly = float(m["weekly_visits"].fillna(0).mean())
            kc1,kc2,kc3,kc4 = st.columns(4)
            kc1.metric("Dealers", kpi_dealers)
            kc2.metric("Active Dealers", int(kpi_active_dealers))
            kc3.metric("Active DSE", kpi_active_dse)
            kc4.metric("Avg Weekly Visits", f"{kpi_avg_weekly:.2f}")

            st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
            center_lon = float(cluster_center["longitude"].mean()) if not cluster_center.empty else float(dealer_rec["longitude"].mean())
            center_lat = float(cluster_center["latitude"].mean()) if not cluster_center.empty else float(dealer_rec["latitude"].mean())
            dealer_rec["color"] = dealer_rec["availability"].map({"Potential":[131,201,255,200],"Low Generation":[255,171,171,200],"Deficit":[255,43,43,200]}).apply(lambda x: x if isinstance(x,list) else [200,200,200,180])

            st.pydeck_chart(
                pdk.Deck(
                    map_style=None,
                    initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
                    tooltip={'text':"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}\nNearest End: {nearest_end_date}"},
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
                            get_fill_color="color",
                            id="dealer",
                            pickable=True,
                            auto_highlight=True
                        ),
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=cluster_center,
                            get_position="[longitude,latitude]",
                            get_radius="size",
                            get_fill_color=[200, 30, 0, 90]
                        ),
                    ]
                )
            )

            st.markdown("<h2 style='font-size:24px;margin:8px 0'>Insights</h2>", unsafe_allow_html=True)
            df_output = dealer_rec[['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']].copy()
            bar_src = df_output[['brand','tag','city']].groupby(['brand','tag']).count().reset_index().rename(columns={'tag':'Penetration','city':'Count Dealers','brand':'Brand'}).sort_values('Penetration')
            fig = px.bar(bar_src, x='Brand', y='Count Dealers', hover_data=['Brand','Count Dealers'], color='Penetration', color_discrete_map={'Not Penetrated':"#83c9ff",'Not Active':'#ffabab','Active':"#ff2b2b"})
            fig.update_layout(legend=dict(orientation='h',yanchor="bottom",y=1.02,xanchor="right",x=1))
            pot_df_output = df_output[['availability','brand','city']].groupby(['availability','brand']).count().reset_index()
            pot_df_output.rename(columns={'availability':'Availability','brand':'Brand','city':'Total Dealers'},inplace=True)
            pie_src = df_output["availability"].value_counts().reset_index()
            pie_src.columns = ["Availability","Total"]
            fig_pie = px.pie(pie_src, names="Availability", values="Total")

            c1,c2 = st.columns([2,1])
            with c1:
                st.plotly_chart(fig, key="bar_penetration", use_container_width=True)
            with c2:
                st.plotly_chart(fig_pie, key="pie_avail", use_container_width=True)

            st.markdown("<h2 style='font-size:24px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
            df_table = df_output.rename(columns={'brand':'Brand','name':'Dealer Name','city':'City','tag':'Activity','joined_dse':'Total Joined DSE','active_dse':'Total Active DSE','nearest_end_date':'Nearest Package End Date','availability':'Availability'})
            df_table = df_table.drop_duplicates(subset=["Dealer Name","City","Brand"])
            st.dataframe(df_table.reset_index(drop=True), use_container_width=True)
