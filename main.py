import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocess import sum_df, clust_df, avail_df_merge, df_visits, jabodetabek_list
import streamlit as st
import pydeck as pdk
from pydeck.types import String

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")
st.markdown("<style>h1{font-size:40px;} h2{font-size:24px;} div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

sales_all = sorted(sum_df["sales_name"].dropna().astype(str).unique().tolist())
areas_all = sorted(avail_df_merge["cluster"].dropna().astype(str).unique().tolist())
brands_all = sorted(avail_df_merge["brand"].dropna().astype(str).unique().tolist())

with st.container(border=True):
    left, right = st.columns(2)
    with left:
        bde = st.selectbox("BDE Name", ["All"] + sales_all, index=0)
        radius = st.slider("Choose Radius (km)", 0, 50, 15)
        penetrated = st.multiselect("Dealer Activity", ["All","Not Active","Not Penetrated","Active"], default=["All"])
    with right:
        area_pick = st.multiselect("Area", ["All"] + areas_all, default=["All"])
        potential = st.multiselect("Dealer Availability", ["All","Potential","Low Generation","Deficit"], default=["All"])
        city_all = sorted(avail_df_merge["city"].dropna().astype(str).unique().tolist())
        city_pick = st.multiselect("Choose City", ["All"] + city_all, default=["All"])
    brand_choose = sorted(avail_df_merge["brand"].dropna().astype(str).unique().tolist())
    brand = st.multiselect("Choose Brand", ["All"] + brand_choose, default=["All"])
    button = st.button("Submit")

def _expand_all(values, universe):
    if "All" in values or not values:
        return universe
    return values

if button:
    p_sel = _expand_all(penetrated, ["Not Active","Not Penetrated","Active"])
    pot_sel = _expand_all(potential, ["Potential","Low Generation","Deficit"])
    area_sel = _expand_all(area_pick, areas_all)
    city_sel = _expand_all(city_pick, city_all)
    brand_sel = _expand_all(brand, brand_choose)

    if bde != "All":
        clusters = clust_df[clust_df["sales_name"] == bde]["cluster"].astype(int).unique().tolist()
        pick_list = []
        for i in clusters:
            dc = f"dist_center_{i}"
            if dc in avail_df_merge.columns:
                tmp = avail_df_merge[(avail_df_merge[dc].notna()) & (avail_df_merge[dc] <= radius) & (avail_df_merge["sales_name"] == bde)].copy()
                if not tmp.empty:
                    tmp["cluster_labels"] = i
                    pick_list.append(tmp)
        pick_avail = pd.concat(pick_list, ignore_index=True) if pick_list else pd.DataFrame()
    else:
        pick_list = []
        for i in range(0, 20):
            dc = f"dist_center_{i}"
            if dc in avail_df_merge.columns:
                tmp = avail_df_merge[avail_df_merge[dc].notna() & (avail_df_merge[dc] <= radius)].copy()
                if not tmp.empty:
                    tmp["cluster_labels"] = i
                    pick_list.append(tmp)
        pick_avail = pd.concat(pick_list, ignore_index=True) if pick_list else pd.DataFrame()

    if pick_avail.empty:
        st.info(f"No dealers found within {radius} km.")
    else:
        drop_cols = [c for c in pick_avail.columns if c.startswith("dist_center_")]
        if drop_cols:
            pick_avail = pick_avail.drop(columns=drop_cols, errors="ignore")
        pick_avail["joined_dse"] = pick_avail["joined_dse"].fillna(0)
        pick_avail["active_dse"] = pick_avail["active_dse"].fillna(0)
        pick_avail["tag"] = np.where((pick_avail["joined_dse"]==0) & (pick_avail["active_dse"]==0), "Not Penetrated", pick_avail["tag"])
        pick_avail["nearest_end_date"] = pick_avail["nearest_end_date"].astype(str)
        pick_avail["nearest_end_date"] = np.where(pick_avail["nearest_end_date"]=="NaT","No Package Found",pick_avail["nearest_end_date"])

        pick_avail = pick_avail[pick_avail["business_type"].astype(str).str.title()=="Car"]

        pick_avail_filter = pick_avail[
            pick_avail["cluster"].astype(str).isin(area_sel) &
            pick_avail["city"].astype(str).isin(city_sel) &
            pick_avail["availability"].astype(str).isin(pot_sel) &
            pick_avail["tag"].astype(str).isin(p_sel) &
            pick_avail["brand"].astype(str).isin(brand_sel)
        ]

        if pick_avail_filter.empty:
            st.info("No dealers match the filters.")
        else:
            if bde != "All":
                cc = clust_df[clust_df["sales_name"] == bde].copy()
            else:
                cc = clust_df.copy()
            if cc.empty:
                st.info("No cluster centers available.")
            else:
                vcount = sum_df if bde == "All" else sum_df[sum_df["sales_name"] == bde]
                cv = vcount[["cluster","sales_name"]].groupby("cluster").count().reset_index().rename(columns={"sales_name":"count_visit"})
                cc = pd.merge(cc, cv, on="cluster", how="left")
                if cc["count_visit"].notna().sum() == 0:
                    cc["count_visit"] = 1
                total_visits = max(cc["count_visit"].sum(), 1)
                cc["size"] = cc["count_visit"]/total_visits*9000
                cc["area_tag"] = cc["cluster"].astype(int) + 1
                cc["word"] = "Area " + cc["area_tag"].astype(str) + "\nCount Visit: " + cc["count_visit"].fillna(0).astype(int).astype(str)
                cc["word_pick"] = "Area " + cc["area_tag"].astype(str)

                dealer_rec = pick_avail_filter.copy()
                dealer_rec["area_tag"] = dealer_rec["cluster_labels"].astype(int) + 1
                dealer_rec["area_tag_word"] = "Area " + dealer_rec["area_tag"].astype(str)
                dealer_rec["penetration_state"] = np.where(dealer_rec["tag"]=="Active","Active", np.where(dealer_rec["tag"]=="Not Penetrated","Not Penetrated","Not Active"))
                color_map = {"Active":[21,255,87,200], "Not Penetrated":[255,43,43,200], "Not Active":[255,171,171,200]}
                dealer_rec["color"] = dealer_rec["penetration_state"].map(color_map)
                dealer_rec["color"] = dealer_rec["color"].apply(lambda x: x if isinstance(x,list) else [200,200,200,180])

                center_lon = float(dealer_rec["longitude"].mean())
                center_lat = float(dealer_rec["latitude"].mean())

                st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
                st.pydeck_chart(
                    pdk.Deck(
                        map_style=None,
                        initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
                        tooltip={'text':"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
                        layers=[
                            pdk.Layer(
                                "TextLayer",
                                data=cc,
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
                                get_color="color",
                                id="dealer",
                                pickable=True,
                                auto_highlight=True
                            ),
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=cc,
                                get_position="[longitude,latitude]",
                                get_radius="size",
                                get_color="[200, 30, 0, 90]"
                            ),
                        ]
                    )
                )

                def some_output(area):
                    df_output = dealer_rec[dealer_rec["area_tag_word"] == area][["brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability"]].copy()
                    st.markdown(f"### There are {len(df_output)} dealers in the radius {radius} km")
                    if not df_output.empty:
                        bar_src = df_output[["brand","tag","city"]].groupby(["brand","tag"]).count().reset_index().rename(columns={"tag":" ","city":"Count Dealers","brand":"Brand"}).sort_values(" ")
                        fig = px.bar(bar_src, x="Brand", y="Count Dealers", hover_data=["Brand","Count Dealers"], color=" ", color_discrete_map={"Not Penetrated":"#83c9ff","Not Active":"#ffabab","Active":"#ff2b2b"})
                        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        pot_df_output = df_output[["availability","brand","city"]].groupby(["availability","brand"]).count().reset_index().rename(columns={"availability":"Availability","brand":"Brand","city":"Total Dealers"})
                        if not pot_df_output.empty:
                            fig1 = px.sunburst(pot_df_output, path=["Availability","Brand"], values="Total Dealers", color="Availability", color_discrete_map={"Potential":"#83c9ff","Low Generation":"#ffabab","Deficit":"#ff2b2b"})
                        else:
                            fig1 = None
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
                        df_shown = df_output.rename(columns={"brand":"Brand","name":"Dealer Name","city":"City","tag":"Activity","joined_dse":"Total Joined DSE","active_dse":"Total Active DSE","nearest_end_date":"Nearest Package End Date","availability":"Availability"})
                        st.markdown("### Dealers Details")
                        st.dataframe(df_shown.drop_duplicates(subset=["Dealer Name","City","Brand"]).reset_index(drop=True), key=f"tbl_{area}")

                st.markdown("<h2 style='font-size:24px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
                tab_labels = cc["word_pick"].dropna().astype(str).unique().tolist()
                tab_labels.sort()
                if not tab_labels:
                    st.info("No areas to display.")
                else:
                    tabs = st.tabs(tab_labels)
                    for tab, label in zip(tabs, tab_labels):
                        with tab:
                            some_output(label)
