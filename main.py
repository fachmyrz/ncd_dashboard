import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocess import *
import streamlit as st
from PIL import Image
import pydeck as pdk

icon = Image.open("assets/favicon.png")
st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=icon, layout="wide")
st.markdown("<h1 style='font-size:40px;margin:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Ahlan','Samin Jaya']
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

base_vis = df_visit.copy()
nik_mask = ~base_vis.get("nomor_induk_karyawan", pd.Series([], dtype=str)).astype(str).str.contains("deleted-", case=False, na=False)
div_mask = ~base_vis.get("divisi", pd.Series([], dtype=str)).astype(str).str.fullmatch("Trainer", case=False, na=False)
bde_list = sorted(base_vis.loc[nik_mask & div_mask, "employee_name"].dropna().astype(str).unique().tolist()) if not base_vis.empty else []

with st.container(border=True):
    bde = st.selectbox("BDE Name", ["All"] + bde_list, index=0)
    cols1 = st.columns(2)
    with cols1[0]:
        penetrated = st.multiselect("Dealer Activity", ["All","Not Active","Not Penetrated","Active"], default=["All"])
        if "All" in penetrated:
            penetrated = ["Not Active","Not Penetrated","Active"]
        radius = st.slider("Choose Radius", 0, 50, 15)
    with cols1[1]:
        potential = st.multiselect("Dealer Availability", ["All","Potential","Low Generation","Deficit"], default=["All"])
        if "All" in potential:
            potential = ["Potential","Low Generation","Deficit"]
        if bde in sales_jabo or bde == "All":
            cities = list(avail_df_merge[(avail_df_merge.city.isin(jabodetabek))]["city"].dropna().unique())
            cities = ["All"] + sorted(cities)
            city_pick = st.multiselect("Choose City", cities, default=["All"])
            if "All" in city_pick:
                city_pick = cities[1:]
        else:
            regional = list(avail_df_merge[(~avail_df_merge.city.isin(jabodetabek))]["city"].dropna().unique())
            regional = ["All"] + sorted(regional)
            city_pick = st.multiselect("Choose City", regional, default=["All"])
            if "All" in city_pick:
                city_pick = regional[1:]
    areas = ["All"] + sorted(avail_df_merge["cluster"].dropna().astype(str).unique().tolist())
    area_pick = st.multiselect("Choose Area", areas, default=["All"])
    if "All" in area_pick:
        area_pick = [a for a in areas if a != "All"]
    brands_base = avail_df_merge["brand"].dropna().unique().tolist()
    brand_choose = ["All"] + sorted(brands_base)
    brand = st.multiselect("Choose Brand", brand_choose, default=["All"])
    if "All" in brand:
        brand = brand_choose[1:]
    button = st.button("Submit")

if not button:
    st.info("Set filters and click Submit.")
else:
    with st.spinner("Computing view..."):
        if bde == "All":
            pick_avail_lst = []
            centers_use = pd.DataFrame()
            for nm in sum_df["sales_name"].dropna().unique():
                n_centers = clust_df[clust_df.sales_name == nm]
                if n_centers.empty:
                    continue
                for i in range(len(n_centers)):
                    col = f"dist_center_{i}"
                    temp = avail_df_merge[(avail_df_merge.get(col).notna()) & (avail_df_merge[col] <= radius) & (avail_df_merge.sales_name == nm)].copy() if col in avail_df_merge.columns else pd.DataFrame()
                    if temp.empty:
                        continue
                    temp["cluster_labels"] = i
                    pick_avail_lst.append(temp)
                centers_use = pd.concat([centers_use, n_centers])
            pick_avail = pd.concat(pick_avail_lst, ignore_index=True) if pick_avail_lst else pd.DataFrame(columns=avail_df_merge.columns.tolist() + ["cluster_labels"])
            cluster_center = centers_use.copy()
        else:
            pick_avail_lst = []
            my_centers = clust_df[clust_df.sales_name == bde]
            for i in range(len(my_centers)):
                col = f"dist_center_{i}"
                if col in avail_df_merge.columns:
                    temp = avail_df_merge[(avail_df_merge[col].notna()) & (avail_df_merge[col] <= radius) & (avail_df_merge.sales_name == bde)].copy()
                    if temp.empty:
                        continue
                    temp["cluster_labels"] = i
                    pick_avail_lst.append(temp)
            pick_avail = pd.concat(pick_avail_lst, ignore_index=True) if pick_avail_lst else pd.DataFrame(columns=avail_df_merge.columns.tolist() + ["cluster_labels"])
            cluster_center = my_centers.copy()
        if pick_avail.empty or cluster_center.empty:
            st.info(f"No dealers found within {radius} km.")
        else:
            for c in [f"dist_center_{i}" for i in range(len(sum_df["cluster"].unique()))]:
                if c in pick_avail.columns:
                    pick_avail.drop(columns=[c], inplace=True, errors="ignore")
            pick_avail["joined_dse"] = pick_avail["joined_dse"].fillna(0)
            pick_avail["active_dse"] = pick_avail["active_dse"].fillna(0)
            pick_avail["tag"] = np.where((pick_avail["joined_dse"]==0) & (pick_avail["active_dse"]==0), "Not Penetrated", pick_avail["tag"])
            pick_avail["nearest_end_date"] = pick_avail["nearest_end_date"].astype(str)
            pick_avail["nearest_end_date"] = np.where(pick_avail["nearest_end_date"] == "NaT", "No Package Found", pick_avail["nearest_end_date"])
            if bde in sales_jabo or bde == "All":
                pick_avail = pick_avail[pick_avail["cluster"].astype(str) == "Jabodetabek"]
                cluster_center = cluster_center[cluster_center["cluster"].astype(str) == "Jabodetabek"]
            else:
                pick_avail = pick_avail[pick_avail["cluster"].astype(str) != "Jabodetabek"]
                cluster_center = cluster_center[cluster_center["cluster"].astype(str) != "Jabodetabek"]
            if city_pick:
                pick_avail = pick_avail[pick_avail["city"].isin(city_pick)]
            if potential:
                pick_avail = pick_avail[pick_avail["availability"].isin(potential)]
            if penetrated:
                pick_avail = pick_avail[pick_avail["tag"].isin(penetrated)]
            if brand:
                pick_avail = pick_avail[pick_avail["brand"].isin(brand)]
            if area_pick:
                pick_avail = pick_avail[pick_avail["cluster"].astype(str).isin(area_pick)]
                cluster_center = cluster_center[cluster_center["cluster"].astype(str).isin(area_pick)]
            dealer_rec = pick_avail.copy()
            if dealer_rec.empty:
                st.info(f"No dealers match the filters within {radius} km.")
            else:
                dealer_rec.sort_values(["cluster_labels","delta","latitude","longitude"], ascending=False, inplace=True)
                if bde == "All":
                    cnt = sum_df.groupby(["sales_name","cluster"]).size().reset_index(name="count_visit")
                    cluster_center = cluster_center.merge(cnt, on=["sales_name","cluster"], how="left")
                else:
                    count_visit_cluster = sum_df[sum_df.sales_name == bde][["cluster","sales_name"]].groupby("cluster").count().reset_index().rename(columns={"sales_name":"count_visit"})
                    cluster_center = cluster_center.merge(count_visit_cluster, on="cluster", how="left")
                total_visits = max(cluster_center["count_visit"].fillna(0).sum(), 1)
                cluster_center["size"] = cluster_center["count_visit"].fillna(0)/total_visits*9000
                cluster_center["area_tag"] = cluster_center["cluster"].astype(int) + 1
                cluster_center["word"] = "Area " + cluster_center["area_tag"].astype(str) + "\nCount Visit: " + cluster_center["count_visit"].fillna(0).astype(int).astype(str)
                cluster_center["word_pick"] = "Area " + cluster_center["area_tag"].astype(str)
                dealer_rec["area_tag"] = dealer_rec["cluster_labels"].astype(int) + 1
                dealer_rec["area_tag_word"] = "Area " + dealer_rec["area_tag"].astype(str)
                dealer_rec["engagement_bucket"] = np.select(
                    [
                        (dealer_rec["tag"]=="Active") & (dealer_rec["active_dse"].fillna(0)>0),
                        (dealer_rec["tag"]=="Not Active") & (dealer_rec["joined_dse"].fillna(0)>0),
                        (dealer_rec["tag"]=="Not Penetrated")
                    ],
                    ["Active","Churn Risk","Not Penetrated"],
                    default="Unknown"
                )
                color_map = {"Active":[21,255,87,220],"Churn Risk":[255,165,0,220],"Not Penetrated":[255,43,43,220],"Unknown":[128,128,128,200]}
                dealer_rec["color"] = dealer_rec["engagement_bucket"].map(color_map)
                dealer_rec["color"] = dealer_rec["color"].apply(lambda x: x if isinstance(x, list) else [128,128,128,200])
                if not cluster_center[["longitude","latitude"]].dropna().empty:
                    center_lon = float(cluster_center["longitude"].mean())
                    center_lat = float(cluster_center["latitude"].mean())
                else:
                    center_lon = float(dealer_rec["longitude"].mean())
                    center_lat = float(dealer_rec["latitude"].mean())
                st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
                st.pydeck_chart(
                    pdk.Deck(
                        map_style=None,
                        initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
                        tooltip={"text":"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
                        layers=[
                            pdk.Layer(
                                "TextLayer",
                                data=cluster_center,
                                get_position="[longitude, latitude]",
                                get_text="word",
                                get_size=12,
                                get_color=[0, 100, 0],
                                get_angle=0,
                                get_text_anchor="middle",
                                get_alignment_baseline="center"
                            ),
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=dealer_rec,
                                get_position="[longitude, latitude]",
                                get_radius=200,
                                get_fill_color="color",
                                id="dealer",
                                pickable=True,
                                auto_highlight=True
                            ),
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=cluster_center,
                                get_position="[longitude, latitude]",
                                get_radius="size",
                                get_fill_color=[200, 30, 0, 90]
                            )
                        ],
                    )
                )
                st.markdown("<h2 style='font-size:24px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
                tab_labels = cluster_center["word_pick"].dropna().unique().tolist()
                tab_labels = sorted(tab_labels, key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 9999)
                def some_output(area, k):
                    df_output = dealer_rec[dealer_rec["area_tag_word"] == area][["brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability"]].copy()
                    st.markdown(f"<h3 style='font-size:18px;margin:4px 0'>There are {len(df_output)} dealers in the radius {radius} km</h3>", unsafe_allow_html=True)
                    if df_output.empty:
                        return
                    bar_src = df_output[["brand","tag","city"]].groupby(["brand","tag"]).count().reset_index().rename(columns={"tag":"Status","city":"Count Dealers","brand":"Brand"}).sort_values("Status")
                    fig = px.bar(bar_src, x="Brand", y="Count Dealers", hover_data=["Brand","Count Dealers"], color="Status", color_discrete_map={"Not Penetrated":"#83c9ff","Not Active":"#ffabab","Active":"#ff2b2b"})
                    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    pot_df = df_output[["availability","brand","city"]].groupby(["availability","brand"]).count().reset_index().rename(columns={"availability":"Availability","brand":"Brand","city":"Total Dealers"})
                    if pot_df.empty:
                        fig1 = go.Figure()
                    else:
                        agg = pot_df.groupby("Availability")["Total Dealers"].sum().reset_index()
                        fig1 = px.pie(agg, names="Availability", values="Total Dealers", hole=0.35, color="Availability", color_discrete_map={"Potential":"#83c9ff","Low Generation":"#ffabab","Deficit":"#ff2b2b"})
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.markdown("<h4 style='font-size:16px;margin:4px 0'>Dealer Penetration</h4>", unsafe_allow_html=True)
                        st.plotly_chart(fig, key=f"bar_{k}", use_container_width=True)
                    with col2:
                        st.markdown("<h4 style='font-size:16px;margin:4px 0'>Potential Dealer</h4>", unsafe_allow_html=True)
                        st.plotly_chart(fig1, key=f"pie_{k}", use_container_width=True)
                    df_shown = df_output.rename(columns={"brand":"Brand","name":"Dealer Name","city":"City","tag":"Activity","joined_dse":"Total Joined DSE","active_dse":"Total Active DSE","nearest_end_date":"Nearest Package End Date","availability":"Availability"}).drop_duplicates(subset=["Dealer Name","City","Brand"])
                    st.dataframe(df_shown.reset_index(drop=True), use_container_width=True, height=420)
                if not tab_labels:
                    st.info("No areas to display.")
                else:
                    tabs = st.tabs(tab_labels)
                    for i, (tab, area_label) in enumerate(zip(tabs, tab_labels)):
                        with tab:
                            some_output(area_label, i)
