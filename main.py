import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import pydeck as pdk
from pydeck.types import String
from data_preprocess import load_sources, compute_all

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")
st.markdown("<h1 style='font-size:40px;margin:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)

with st.spinner("Loading sources..."):
    dealers, visits, ro, ld, nc = load_sources()

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Ahlan','Samin Jaya']
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

with st.container(border=True):
    base = visits.copy()
    nik_mask = ~base.get("Nomor Induk Karyawan", pd.Series([], dtype=str)).astype(str).str.contains("deleted-", case=False, na=False)
    div_mask = ~base.get("Divisi", pd.Series([], dtype=str)).astype(str).str.contains("trainer", case=False, na=False)
    bde_list = sorted(base.loc[nik_mask & div_mask, "employee_name"].dropna().astype(str).unique().tolist())
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
        areas = sorted(ld["cluster"].dropna().astype(str).unique().tolist())
        area_pick = st.multiselect("Choose Area", ["All"] + areas, default=["All"])
        if "All" in area_pick:
            area_pick = areas
        if name in sales_jabo or name == "All":
            cities = list(dealers[(dealers.city.isin(jabodetabek))]["city"].dropna().unique())
        else:
            cities = list(dealers[(~dealers.city.isin(jabodetabek))]["city"].dropna().unique())
        city_pick = st.multiselect("Choose City", ["All"] + cities, default=["All"])
        if "All" in city_pick:
            city_pick = cities
    brand_choose = sorted(dealers["brand"].dropna().unique().tolist())
    brand = st.multiselect("Choose Brand", ["All"] + brand_choose, default=["All"])
    if "All" in brand:
        brand = brand_choose
    button = st.button("Submit")

if not button:
    st.info("Set filters and click Submit.")
else:
    with st.spinner("Computing view..."):
        sum_df, centers, avail_df_merge, dealer_rec, metrics = compute_all(name, radius, area_pick, city_pick, brand, penetrated, potential)
    if name in sales_jabo or name == "All":
        dealer_rec = dealer_rec[dealer_rec.cluster.astype(str) == "Jabodetabek"]
        centers = centers[centers.cluster.astype(str) == "Jabodetabek"] if not centers.empty else centers
    else:
        dealer_rec = dealer_rec[dealer_rec.cluster.astype(str) != "Jabodetabek"]
        centers = centers[centers.cluster.astype(str) != "Jabodetabek"] if not centers.empty else centers
    if dealer_rec.empty:
        st.info(f"No dealers match filters within {radius} km.")
    else:
        centers = centers.copy()
        centers["count_visit"] = 0
        if name != "All":
            cv = sum_df[sum_df.sales_name == name][["cluster","sales_name"]].groupby("cluster").count().reset_index().rename(columns={"sales_name":"count_visit"})
        else:
            cv = sum_df[["cluster","sales_name"]].groupby("cluster").count().reset_index().rename(columns={"sales_name":"count_visit"})
        centers = pd.merge(centers, cv, on="cluster", how="left")
        centers["count_visit"] = centers["count_visit"].fillna(0)
        total_visits = max(centers["count_visit"].sum(), 1)
        centers["size"] = centers["count_visit"] / total_visits * 9000
        centers["area_tag"] = centers["cluster"].astype(str)
        centers["word"] = "Area " + centers["area_tag"].astype(str) + "\nCount Visit: " + centers["count_visit"].astype(int).astype(str)
        centers["word_pick"] = "Area " + centers["area_tag"].astype(str)
        dealer_rec = dealer_rec.copy()
        dealer_rec["area_tag"] = dealer_rec.get("cluster_labels", pd.Series([0]*len(dealer_rec))).fillna(0).astype(int) + 1
        dealer_rec["area_tag_word"] = "Area " + dealer_rec["area_tag"].astype(str)
        dealer_rec["engagement_bucket"] = np.select(
            [
                dealer_rec["active_dse"].fillna(0) > 0,
                (dealer_rec["joined_dse"].fillna(0) == 0) & (dealer_rec["active_dse"].fillna(0) == 0)
            ],
            ["Active", "Not Penetrated"],
            default="Not Active"
        )
        color_map = {"Active": [46,204,113,200], "Not Penetrated": [255,99,132,200], "Not Active": [255,195,0,200]}
        dealer_rec["color"] = dealer_rec["engagement_bucket"].map(color_map).apply(lambda x: x if isinstance(x, list) else [200,200,200,180])
        center_lon = float(centers.longitude.mean()) if not centers.empty else float(dealer_rec.longitude.mean())
        center_lat = float(centers.latitude.mean()) if not centers.empty else float(dealer_rec.latitude.mean())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dealers", f"{metrics.get('dealers',0)}")
        c2.metric("Active Dealers", f"{metrics.get('active_dealers',0)}")
        c3.metric("Active DSE", f"{metrics.get('active_dse',0)}")
        c4.metric("Avg Weekly Visits", f"{metrics.get('avg_weekly',0):.2f}")
        st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
                tooltip={"text": "Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nStatus: {engagement_bucket}"},
                layers=[
                    pdk.Layer(
                        "TextLayer",
                        data=centers,
                        get_position="[longitude,latitude]",
                        get_text="word",
                        get_size=12,
                        get_color=[0,100,0],
                        get_angle=0,
                        get_text_anchor=String("middle"),
                        get_alignment_baseline=String("center"),
                        id="txt_layer"
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=dealer_rec,
                        get_position="[longitude,latitude]",
                        get_radius=200,
                        get_color="color",
                        id="dealer_layer",
                        pickable=True,
                        auto_highlight=True
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=centers,
                        get_position="[longitude,latitude]",
                        get_radius="size",
                        get_color="[200, 30, 0, 90]",
                        id="center_layer"
                    ),
                ]
            ),
            key=f"deck_{name}_{radius}_{len(dealer_rec)}"
        )
        def some_output(area):
            df_output = dealer_rec[dealer_rec.area_tag_word == area][["brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability","avg_weekly_visits"]]
            st.markdown(f"<h3 style='font-size:20px;margin:8px 0'>There are {len(df_output)} dealers in the radius {radius} km</h3>", unsafe_allow_html=True)
            if not df_output.empty:
                bar_src = df_output[["brand","tag","city"]].groupby(["brand","tag"]).count().reset_index().rename(columns={"tag":" ","city":"Count Dealers","brand":"Brand"}).sort_values(" ")
                fig = px.bar(bar_src, x="Brand", y="Count Dealers", hover_data=["Brand","Count Dealers"], color=" ", color_discrete_map={"Not Penetrated":"#83c9ff","Not Active":"#ffabab","Active":"#ff2b2b"})
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                pot_df_output = df_output[["availability","brand","city"]].groupby(["availability","brand"]).count().reset_index()
                pot_df_output.rename(columns={"availability":"Availability","brand":"Brand","city":"Total Dealers"}, inplace=True)
                fig1 = px.sunburst(pot_df_output, path=["Availability","Brand"], values="Total Dealers", color="Availability", color_discrete_map={"Potential":"#83c9ff","Low Generation":"#ffabab","Deficit":"#ff2b2b"}) if not pot_df_output.empty else None
            else:
                fig = None
                fig1 = None
            col1, col2 = st.columns([2,1])
            with col1:
                if fig is not None:
                    st.markdown("#### Dealer Penetration")
                    st.plotly_chart(fig, key=f"bar_{area}")
            with col2:
                if fig1 is not None:
                    st.markdown("#### Potential Dealer")
                    st.plotly_chart(fig1, key=f"sun_{area}")
            if not df_output.empty:
                df_shown = df_output.rename(columns={"brand":"Brand","name":"Dealer Name","city":"City","tag":"Activity","joined_dse":"Total Joined DSE","active_dse":"Total Active DSE","nearest_end_date":"Nearest Package End Date","availability":"Availability","avg_weekly_visits":"Avg Weekly Visits"})
                st.markdown("### Dealers Details")
                st.dataframe(df_shown.drop_duplicates(subset=["Dealer Name"]).reset_index(drop=True), key=f"tbl_{area}")
        st.markdown("<h2 style='font-size:24px;margin:12px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
        tab_labels = centers["word_pick"] if "word_pick" in centers.columns else ("Area " + centers["cluster"].astype(str))
        tab_labels = tab_labels.dropna().unique().tolist()
        tab_labels.sort()
        tabs = st.tabs(tab_labels if tab_labels else ["No Area"])
        for tab, area_label in zip(tabs, tab_labels if tab_labels else ["No Area"]):
            with tab:
                if area_label == "No Area":
                    st.info("No areas to display.")
                else:
                    some_output(area_label)
