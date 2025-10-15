import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pydeck as pdk
from data_preprocess import avail_df_merge, df_visit

icon = Image.open("assets/favicon.png")
st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=icon, layout="wide")

st.markdown("<style>h1, .st-emotion-cache-10trblm {font-size:40px !important}</style>", unsafe_allow_html=True)
st.title("Dealer Penetration Dashboard")

df = avail_df_merge.copy()
df = df.dropna(subset=["latitude","longitude"])
df = df[~df["latitude"].isna() & ~df["longitude"].isna()]
df = df.drop_duplicates(subset=["id_dealer_outlet"])

areas = ["Jabodetabek","Regional","All"]
bde_list = sorted([x for x in df_visit.get("employee_name", pd.Series([], dtype=str)).dropna().unique().tolist() if x])
brands = sorted([x for x in df.get("brand", pd.Series([], dtype=str)).dropna().unique().tolist() if x])
cities_all = sorted([x for x in df.get("city", pd.Series([], dtype=str)).dropna().unique().tolist() if x])

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.selectbox("Area", areas, index=0)
    with c2:
        bde = st.multiselect("BDE Name", ["All"] + bde_list, default=["All"])
    with c3:
        radius = st.slider("Radius (km) for insights", 0, 50, 15)
    c4, c5, c6 = st.columns(3)
    with c4:
        activity = st.multiselect("Dealer Activity", ["All","Active","Not Active","Not Penetrated"], default=["All"])
    with c5:
        availability = st.multiselect("Dealer Availability", ["All","Potential","Low Generation","Deficit"], default=["All"])
    with c6:
        if area == "Jabodetabek":
            cities = sorted(df.loc[df["cluster"]=="Jabodetabek","city"].dropna().unique().tolist())
        elif area == "Regional":
            cities = sorted(df.loc[df["cluster"]!="Jabodetabek","city"].dropna().unique().tolist())
        else:
            cities = cities_all
        city_pick = st.multiselect("City", ["All"] + cities, default=["All"])
    brand = st.multiselect("Brand", ["All"] + brands, default=["All"])
    button = st.button("Apply")

if button:
    dff = df.copy()
    if area != "All":
        dff = dff[dff["cluster"].fillna("Regional").eq(area)]
    if "All" not in bde:
        last_by = dff.get("last_visited_by")
        if last_by is not None:
            dff = dff[last_by.isin(bde)]
        else:
            dff = dff.iloc[0:0]
    if "All" not in activity:
        dff = dff[dff["tag"].isin(activity)]
    if "All" not in availability:
        dff = dff[dff["availability"].isin(availability)]
    if "All" not in city_pick:
        dff = dff[dff["city"].isin(city_pick)]
    if "All" not in brand:
        dff = dff[dff["brand"].isin(brand)]

    total_dealers = int(dff["id_dealer_outlet"].nunique())
    active_dealers = int(dff.loc[dff["active_dse"]>0,"id_dealer_outlet"].nunique())
    active_dse = int(dff["active_dse"].fillna(0).sum())
    avg_weekly = round(dff["avg_weekly_visits"].fillna(0).mean(), 2) if not dff.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dealers", f"{total_dealers}")
    k2.metric("Active Dealers", f"{active_dealers}")
    k3.metric("Active DSE", f"{active_dse}")
    k4.metric("Avg Weekly Visits", f"{avg_weekly}")

    if dff.empty:
        st.info("No data for the selected filters.")
    else:
        center_lon = float(dff["longitude"].astype(float).mean())
        center_lat = float(dff["latitude"].astype(float).mean())

        def engagement_bucket(r):
            if r.get("active_dse",0) > 0:
                return "Active"
            if r.get("visits_last_N",0) == 0 and r.get("joined_dse",0) == 0:
                return "Not Penetrated"
            if r.get("visits_last_N",0) == 0:
                return "Not Active"
            return "Active"

        dff["engagement_bucket"] = dff.apply(engagement_bucket, axis=1)
        color_map = {
            "Active":[21,255,87,200],
            "Not Active":[255,171,171,200],
            "Not Penetrated":[131,201,255,200]
        }
        dff["color"] = dff["engagement_bucket"].map(color_map)
        dff["color"] = dff["color"].apply(lambda x: x if isinstance(x, list) else [200,200,200,180])

        st.subheader("Penetration Map")
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    longitude=center_lon,
                    latitude=center_lat,
                    zoom=10 if area=="Jabodetabek" else 7,
                    pitch=45
                ),
                tooltip={"text":"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nActivity: {engagement_bucket}\nNearest End: {nearest_end_date}"},
                layers=[
                    pdk.Layer(
                        "HexagonLayer",
                        data=dff,
                        get_position="[longitude, latitude]",
                        radius=2000 if area!="Jabodetabek" else 1200,
                        elevation_scale=4,
                        elevation_range=[0,1000],
                        pickable=True,
                        extruded=True
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=dff,
                        get_position="[longitude, latitude]",
                        get_radius=200,
                        get_color="color",
                        pickable=True,
                        auto_highlight=True,
                        id="dealer"
                    )
                ]
            )
        )

        st.subheader("Dealers Detail")
        table = dff[[
            "id_dealer_outlet","client_name","name","brand","city","cluster","availability","engagement_bucket","joined_dse","active_dse","total_dse","avg_weekly_visits","last_visit_datetime","last_visited_by","nearest_end_date"
        ]].copy()
        table = table.rename(columns={"client_name":"dealer_name","engagement_bucket":"activity"})
        table = table.sort_values(["activity","availability","city","brand","dealer_name"]).reset_index(drop=True)
        st.dataframe(table, use_container_width=True)
