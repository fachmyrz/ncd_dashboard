import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocess import avail_df_merge, sum_df, clust_df, df_visit
import streamlit as st
from PIL import Image
import pydeck as pdk
from pydeck.types import String

icon = Image.open("assets/favicon.png")
st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=icon, layout="wide")

sales_jabo = ["A. Sofyan","Nova Handoyo","Heriyanto","Aditya rifat","Riski Amrullah Zulkarnain","Rudy Setya Wibowo","Muhammad Ahlan","Samin Jaya"]
jabodetabek = ["Bekasi","Bogor","Depok","Jakarta Barat","Jakarta Pusat","Jakarta Selatan","Jakarta Timur","Jakarta Utara","Tangerang","Tangerang Selatan","Cibitung","Tambun","Cikarang","Karawaci","Alam Sutera","Cileungsi","Sentul","Cibubur","Bintaro"]

st.markdown("<h1 style='font-size:40px;margin-top:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)

with st.container(border=True):
    bde_base = df_visit.copy()
    mask = ~bde_base.get("employee_nik", pd.Series([], dtype=str)).astype(str).str.contains("deleted-", case=False, na=False)
    mask &= ~bde_base.get("division", pd.Series([], dtype=str)).astype(str).str.contains("Trainer", case=False, na=False)
    bde_options = sorted(bde_base.loc[mask, "employee_name"].dropna().unique().tolist())
    bde_options = ["All"] + bde_options
    name = st.selectbox("BDE Name", bde_options, index=0)

    col_area, col_city = st.columns(2)
    with col_area:
        area_opt = ["Jabodetabek","Non-Jabodetabek"]
        area_pick = st.selectbox("Area", area_opt, index=0)
    with col_city:
        if area_pick == "Jabodetabek":
            base_city = jabodetabek
        else:
            base_city = sorted(list(set(avail_df_merge["city"].dropna().unique().tolist()) - set(jabodetabek)))
        city_list = ["All"] + base_city
        city_pick = st.multiselect("Choose City", city_list, default=["All"])

    col1, col2 = st.columns(2)
    with col1:
        penetrated = st.multiselect("Dealer Activity", ["All","Not Active","Not Penetrated","Active"], default=["All"])
        if "All" in penetrated:
            penetrated = ["Not Active","Not Penetrated","Active"]
        radius = st.slider("Choose Radius (km)", 0, 50, 15)
    with col2:
        potential = st.multiselect("Dealer Availability", ["All","Potential","Low Generation","Deficit"], default=["All"])
        if "All" in potential:
            potential = ["Potential","Low Generation","Deficit"]
        if name in sales_jabo or name == "All":
            brand_pool = avail_df_merge[avail_df_merge["city"].isin(jabodetabek)]
        else:
            brand_pool = avail_df_merge[~avail_df_merge["city"].isin(jabodetabek)]
        brand_choose = sorted(brand_pool["brand"].dropna().unique().tolist())
        brand_choose = ["All"] + brand_choose
        brand = st.multiselect("Choose Brand", brand_choose, default=["All"])
        if "All" in brand:
            brand = brand_choose[1:]

    button = st.button("Submit")

if button:
    if name == "All":
        names_in_scope = sum_df["sales_name"].dropna().unique().tolist()
    else:
        names_in_scope = [name]

    pick_avail_lst = []
    for nm in names_in_scope:
        clusters = sum_df[sum_df["sales_name"] == nm]["cluster"].dropna().unique().tolist()
        for i in clusters:
            col = f"dist_center_{int(i)}"
            subset = avail_df_merge[(avail_df_merge.get(col).notna()) & (avail_df_merge[col] <= radius)]
            subset = subset[subset["sales_name"] == nm]
            if not subset.empty:
                subset = subset.copy()
                subset["cluster_labels"] = int(i)
                pick_avail_lst.append(subset)

    if not pick_avail_lst:
        st.info(f"No dealers found within {radius} km.")
        st.stop()

    pick_avail = pd.concat(pick_avail_lst, ignore_index=True)
    drop_cols = [c for c in pick_avail.columns if c.startswith("dist_center_")]
    if drop_cols:
        pick_avail = pick_avail.drop(columns=drop_cols, errors="ignore")

    pick_avail["joined_dse"] = pick_avail["joined_dse"].fillna(0).astype(int)
    pick_avail["active_dse"] = pick_avail["active_dse"].fillna(0).astype(int)
    pick_avail["tag"] = np.where((pick_avail["joined_dse"] == 0) & (pick_avail["active_dse"] == 0), "Not Penetrated", pick_avail["tag"])
    pick_avail["nearest_end_date"] = pd.to_datetime(pick_avail["nearest_end_date"], errors="coerce").dt.date

    if area_pick == "Jabodetabek":
        pick_avail = pick_avail[pick_avail["city"].isin(jabodetabek)]
    else:
        pick_avail = pick_avail[~pick_avail["city"].isin(jabodetabek)]

    if "All" in city_pick:
        city_scope = base_city
    else:
        city_scope = city_pick

    pick_avail_filter = pick_avail[
        pick_avail["city"].isin(city_scope)
        & pick_avail["availability"].isin(potential)
        & pick_avail["tag"].isin(penetrated)
        & pick_avail["brand"].isin(brand)
    ].copy()

    if pick_avail_filter.empty:
        st.info("No dealers match the filters.")
        st.stop()

    dealer_rec = pick_avail_filter.copy()
    dealer_rec["cluster_labels"] = dealer_rec["cluster_labels"].fillna(0).astype(int)
    dealer_rec["area_tag"] = dealer_rec["cluster_labels"].astype(int) + 1
    dealer_rec["area_tag_word"] = "Area " + dealer_rec["area_tag"].astype(str)

    if name == "All":
        cluster_center = clust_df[clust_df["sales_name"].isin(names_in_scope)].copy()
    else:
        cluster_center = clust_df[clust_df["sales_name"] == name].copy()
    count_visit_cluster = sum_df[sum_df["sales_name"].isin(names_in_scope)][["cluster","sales_name"]].groupby("cluster").count().reset_index().rename(columns={"sales_name":"count_visit"})
    cluster_center = pd.merge(cluster_center, count_visit_cluster, on="cluster", how="left").fillna({"count_visit":0})
    total_visits = max(cluster_center["count_visit"].sum(), 1)
    cluster_center["size"] = cluster_center["count_visit"] / total_visits * 9000
    cluster_center["area_tag"] = cluster_center["cluster"].astype(int) + 1
    cluster_center["word"] = "Area " + cluster_center["area_tag"].astype(str) + "\nCount Visit: " + cluster_center["count_visit"].astype(int).astype(str)
    cluster_center["word_pick"] = "Area " + cluster_center["area_tag"].astype(str)

    center_lon = float(dealer_rec["longitude"].mean()) if dealer_rec["longitude"].notna().any() else 106.8456
    center_lat = float(dealer_rec["latitude"].mean()) if dealer_rec["latitude"].notna().any() else -6.2088

    dealer_rec["engagement_bucket"] = np.select(
        [
            (dealer_rec["active_dse"] > 0),
            (dealer_rec["joined_dse"] > 0) & (dealer_rec["active_dse"] == 0),
            (dealer_rec["tag"] == "Not Penetrated"),
        ],
        ["Active", "Joined Not Active", "Not Penetrated"],
        default="Unknown",
    )

    color_map = {
        "Active":[21, 255, 87, 200],
        "Joined Not Active":[255, 210, 0, 200],
        "Not Penetrated":[255, 43, 43, 200],
        "Unknown":[180, 180, 180, 180],
    }
    dealer_rec["color"] = dealer_rec["engagement_bucket"].map(color_map)
    dealer_rec["color"] = dealer_rec["color"].apply(lambda x: x if isinstance(x, list) else [180,180,180,180])

    st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
            tooltip={"text":"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nStatus: {engagement_bucket}"},
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=dealer_rec,
                    get_position="[longitude,latitude]",
                    get_radius=220,
                    get_fill_color="color",
                    id="dealer_pts",
                    pickable=True,
                    auto_highlight=True,
                ),
                pdk.Layer(
                    "TextLayer",
                    data=cluster_center,
                    get_position="[longitude,latitude]",
                    get_text="word",
                    get_size=12,
                    get_color=[0, 100, 0],
                    get_angle=0,
                    get_text_anchor=String("middle"),
                    get_alignment_baseline=String("center"),
                    id="area_labels",
                ),
            ],
        )
    )

    visits_clean = df_visit.dropna(subset=["client_name","date"]).copy()
    weekly = (
        visits_clean.assign(week=pd.to_datetime(visits_clean["date"]).astype("datetime64[ns]").dt.to_period("W").astype(str))
        .groupby(["client_name","week"], as_index=False)
        .size()
        .groupby("client_name", as_index=False)["size"].mean()
        .rename(columns={"client_name":"name","size":"avg_weekly_visits"})
    )
    dealer_rec = dealer_rec.merge(weekly, on="name", how="left")
    dealer_rec["avg_weekly_visits"] = dealer_rec["avg_weekly_visits"].fillna(0).round(2)

    total_dealers = int(dealer_rec["id_dealer_outlet"].nunique())
    active_dealers = int(dealer_rec.loc[dealer_rec["active_dse"] > 0, "id_dealer_outlet"].nunique())
    active_dse = int(dealer_rec["active_dse"].fillna(0).sum())
    avg_weekly = float(dealer_rec["avg_weekly_visits"].mean()) if not dealer_rec.empty else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Dealers", f"{total_dealers}")
    with m2:
        st.metric("Active Dealers", f"{active_dealers}")
    with m3:
        st.metric("Active DSE", f"{active_dse}")
    with m4:
        st.metric("Avg Weekly Visits", f"{avg_weekly:.2f}")

    city_pen = dealer_rec.groupby("city", as_index=False).agg(
        dealers=("id_dealer_outlet","nunique"),
        active=("active_dse", lambda s: (s > 0).sum()),
        not_pen=("engagement_bucket", lambda s: (s == "Not Penetrated").sum()),
        potential=("availability", lambda s: (s == "Potential").sum()),
    )
    city_pen["not_pen_rate"] = (city_pen["not_pen"] / city_pen["dealers"]).fillna(0)
    city_pen["priority"] = np.where(city_pen["not_pen_rate"] >= 0.5, "High", np.where(city_pen["not_pen_rate"] >= 0.25, "Medium", "Low"))
    pri_map = {"High":[255, 43, 43, 120], "Medium":[255, 210, 0, 120], "Low":[21, 255, 87, 120]}
    city_centers = dealer_rec.groupby("city", as_index=False)[["latitude","longitude"]].mean()
    city_pen = city_pen.merge(city_centers, on="city", how="left")
    city_pen["color"] = city_pen["priority"].map(pri_map)

    st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Highlight</h2>", unsafe_allow_html=True)
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=9, pitch=0),
            tooltip={"text":"City: {city}\nDealers: {dealers}\nNot Penetrated: {not_pen}\nPriority: {priority}"},
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=city_pen.dropna(subset=["latitude","longitude"]),
                    get_position="[longitude,latitude]",
                    get_radius=1500,
                    get_fill_color="color",
                    id="city_highlight",
                    pickable=True,
                )
            ],
        )
    )

    def some_output(area):
        df_output = dealer_rec[dealer_rec["area_tag_word"] == area][["brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability","avg_weekly_visits"]].copy()
        st.markdown(f"### {len(df_output)} dealers in {area} within {radius} km")
        if not df_output.empty:
            bar_src = df_output.groupby(["brand","tag"], as_index=False).size().rename(columns={"size":"Count Dealers","brand":"Brand","tag":" "})
            fig = px.bar(bar_src, x="Brand", y="Count Dealers", color=" ", hover_data=["Brand","Count Dealers"])
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        else:
            fig = None
        col1, col2 = st.columns([2,1])
        with col1:
            if fig is not None:
                st.markdown("#### Dealer Penetration")
                st.plotly_chart(fig, key=f"bar_{area}")
        with col2:
            st.markdown("#### Key Stats")
            st.write(df_output.agg({"joined_dse":"sum","active_dse":"sum","avg_weekly_visits":"mean"}).round(2).to_frame("value"))
        if not df_output.empty:
            df_shown = df_output.rename(columns={
                "brand":"Brand",
                "name":"Dealer",
                "city":"City",
                "tag":"Activity",
                "joined_dse":"Total Joined DSE",
                "active_dse":"Total Active DSE",
                "nearest_end_date":"Nearest Package End Date",
                "availability":"Availability",
                "avg_weekly_visits":"Avg Weekly Visits"
            }).drop_duplicates(subset=["Dealer"])
            st.markdown("### Dealers Details")
            st.dataframe(df_shown.reset_index(drop=True), use_container_width=True, key=f"tbl_{area}")

    st.markdown("<h2 style='font-size:24px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
    tab_labels = cluster_center["word_pick"].dropna().unique().tolist()
    tab_labels.sort()
    for tab, area_label in zip(st.tabs(tab_labels if tab_labels else ["Overview"]), tab_labels if tab_labels else ["Overview"]):
        with tab:
            if area_label == "Overview":
                st.dataframe(dealer_rec[["name","city","brand","availability","engagement_bucket","joined_dse","active_dse","avg_weekly_visits","nearest_end_date"]].drop_duplicates(subset=["name"]).reset_index(drop=True), use_container_width=True, key="tbl_overview")
            else:
                some_output(area_label)
