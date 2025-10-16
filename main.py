import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocess import sum_df, clust_df, avail_df_merge, df_visits
import streamlit as st
from PIL import Image
import pydeck as pdk
from pydeck.types import String

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Ahlan','Samin Jaya']
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

st.markdown("<h1 style='font-size:40px;margin:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)

with st.container(border=True):
    base_visit = df_visits.copy()
    nik_mask = ~base_visit.get("Nomor Induk Karyawan", pd.Series([], dtype=str)).astype(str).str.contains("deleted-", case=False, na=False)
    div_mask = ~base_visit.get("Divisi", pd.Series([], dtype=str)).astype(str).str.contains("trainer", case=False, na=False)
    bde_list = sorted(base_visit.loc[nik_mask & div_mask, "employee_name"].dropna().astype(str).unique().tolist())
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
        areas = sorted(avail_df_merge["cluster"].dropna().astype(str).unique().tolist())
        area_pick = st.multiselect("Choose Area", ["All"] + areas, default=["All"])
        if "All" in area_pick:
            area_pick = areas
        if name in sales_jabo:
            cities = list(avail_df_merge[(avail_df_merge.city.isin(jabodetabek))]["city"].dropna().unique())
        else:
            cities = list(avail_df_merge[(~avail_df_merge.city.isin(jabodetabek))]["city"].dropna().unique())
        city_pick = st.multiselect("Choose City", ["All"] + cities, default=["All"])
        if "All" in city_pick:
            city_pick = cities
    subdf = avail_df_merge.copy()
    if name != "All":
        subdf = subdf[subdf["sales_name"] == name]
    subdf = subdf[subdf["cluster"].astype(str).isin([str(x) for x in area_pick])] if area_pick else subdf
    subdf = subdf[subdf["city"].isin(city_pick)] if city_pick else subdf
    subdf = subdf[subdf["availability"].isin(potential)] if potential else subdf
    brand_choose = list(subdf[subdf["tag"].isin(penetrated)]["brand"].dropna().unique())
    brand = st.multiselect("Choose Brand", ["All"] + brand_choose, default=["All"])
    if "All" in brand:
        brand = brand_choose
    button = st.button("Submit")

def build_radius_frame(name_sel, base, rad):
    if name_sel == "All":
        names = sum_df["sales_name"].dropna().unique().tolist()
    else:
        names = [name_sel]
    picks = []
    for nm in names:
        ccols = [c for c in base.columns if str(c).startswith("dist_center_")]
        if not ccols:
            continue
        for i in range(len(ccols)):
            col = f"dist_center_{i}"
            if col in base.columns:
                temp = base[(base[col].notna()) & (base[col] <= rad)]
                if name_sel != "All":
                    temp = temp[temp["sales_name"] == nm]
                if not temp.empty:
                    t = temp.copy()
                    t["cluster_labels"] = i
                    picks.append(t)
    if not picks:
        return pd.DataFrame(columns=base.columns.tolist() + ["cluster_labels"])
    out = pd.concat(picks, ignore_index=True)
    drop_cols = [f"dist_center_{i}" for i in range(len([c for c in base.columns if str(c).startswith("dist_center_")])) if f"dist_center_{i}" in out.columns]
    if drop_cols:
        out.drop(columns=drop_cols, inplace=True, errors="ignore")
    return out

if button:
    pick_avail = build_radius_frame(name, avail_df_merge, radius)
    if name in sales_jabo:
        pick_avail = pick_avail[pick_avail.cluster.astype(str) == "Jabodetabek"]
    else:
        pick_avail = pick_avail[pick_avail.cluster.astype(str) != "Jabodetabek"]
    pick_avail["joined_dse"] = pick_avail["joined_dse"].fillna(0)
    pick_avail["active_dse"] = pick_avail["active_dse"].fillna(0)
    pick_avail["tag"] = np.where((pick_avail["joined_dse"] == 0) & (pick_avail["active_dse"] == 0), "Not Penetrated", pick_avail["tag"])
    pick_avail["nearest_end_date"] = pd.to_datetime(pick_avail["nearest_end_date"], errors="coerce").dt.date.astype(str)
    pick_avail["nearest_end_date"] = np.where(pick_avail["nearest_end_date"] == "NaT", "No Package Found", pick_avail["nearest_end_date"])
    if penetrated and potential and city_pick:
        pick_avail_filter = pick_avail[(pick_avail.city.isin(city_pick)) & (pick_avail.availability.isin(potential)) & (pick_avail.tag.isin(penetrated)) & (pick_avail.brand.isin(brand)) & (pick_avail.cluster.astype(str).isin([str(x) for x in area_pick]))]
    else:
        pick_avail_filter = pick_avail
    dealer_rec = pick_avail_filter.copy()
    if dealer_rec.empty:
        st.info(f"No dealers match filters within {radius} km.")
    else:
        dealer_rec.sort_values(["cluster_labels", "delta", "latitude", "longitude"], ascending=False, inplace=True)
        if name == "All":
            centers = clust_df.copy()
            cv = sum_df[["cluster", "sales_name"]].groupby("cluster").count().reset_index().rename(columns={"sales_name": "count_visit"})
            centers = pd.merge(centers, cv, on="cluster", how="left")
        else:
            centers = clust_df[clust_df.sales_name == name].copy()
            cv = sum_df[sum_df.sales_name == name][["cluster", "sales_name"]].groupby("cluster").count().reset_index().rename(columns={"sales_name": "count_visit"})
            centers = pd.merge(centers, cv, on="cluster", how="left")
        centers["count_visit"] = centers["count_visit"].fillna(0)
        total_visits = max(centers["count_visit"].sum(), 1)
        centers["size"] = centers["count_visit"] / total_visits * 9000
        centers["area_tag"] = centers["cluster"].astype(int) + 1
        centers["word"] = "Area " + centers["area_tag"].astype(str) + "\nCount Visit: " + centers["count_visit"].astype(int).astype(str)
        centers["word_pick"] = "Area " + centers["area_tag"].astype(str)
        dealer_rec["area_tag"] = dealer_rec["cluster_labels"].astype(int) + 1
        dealer_rec["area_tag_word"] = "Area " + dealer_rec["area_tag"].astype(str)
        dealer_rec["engagement_bucket"] = np.select(
            [
                dealer_rec["active_dse"].fillna(0) > 0,
                (dealer_rec["joined_dse"].fillna(0) == 0) & (dealer_rec["active_dse"].fillna(0) == 0)
            ],
            ["Active", "Not Penetrated"],
            default="Not Active"
        )
        color_map = {"Active": [46, 204, 113, 200], "Not Penetrated": [255, 99, 132, 200], "Not Active": [255, 195, 0, 200]}
        dealer_rec["color"] = dealer_rec["engagement_bucket"].map(color_map)
        dealer_rec["color"] = dealer_rec["color"].apply(lambda x: x if isinstance(x, list) else [200, 200, 200, 180])
        center_lon = float(centers.longitude.mean()) if not centers.empty else float(dealer_rec.longitude.mean())
        center_lat = float(centers.latitude.mean()) if not centers.empty else float(dealer_rec.latitude.mean())
        m_dealers = int(dealer_rec["id_dealer_outlet"].nunique())
        m_active_dealers = int(dealer_rec[dealer_rec["active_dse"].fillna(0) > 0]["id_dealer_outlet"].nunique())
        m_active_dse = int(dealer_rec["active_dse"].fillna(0).sum())
        m_avg_week = float(dealer_rec["avg_weekly_visits"].fillna(0).mean()) if "avg_weekly_visits" in dealer_rec.columns else 0.0
        cols_m = st.columns(4)
        cols_m[0].metric("Dealers", f"{m_dealers}")
        cols_m[1].metric("Active Dealers", f"{m_active_dealers}")
        cols_m[2].metric("Active DSE", f"{m_active_dse}")
        cols_m[3].metric("Avg Weekly Visits", f"{m_avg_week:.2f}")
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
                        get_color=[0, 100, 0],
                        get_angle=0,
                        get_text_anchor=String("middle"),
                        get_alignment_baseline=String("center"),
                        id="txt"
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
                        data=centers,
                        get_position="[longitude,latitude]",
                        get_radius="size",
                        get_color="[200, 30, 0, 90]",
                        id="center"
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
        tab_labels = centers["word_pick"].dropna().unique().tolist()
        tab_labels.sort()
        tabs = st.tabs(tab_labels if tab_labels else ["No Area"])
        for tab, area_label in zip(tabs, tab_labels if tab_labels else ["No Area"]):
            with tab:
                if area_label == "No Area":
                    st.info("No areas to display.")
                else:
                    some_output(area_label)
