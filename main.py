import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocess import sum_df, clust_df, avail_df_merge, df_visit
import streamlit as st
from PIL import Image
import pydeck as pdk
from pydeck.types import String

icon = None
try:
    icon = Image.open("assets/favicon.png")
except:
    icon = None

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon=icon if icon is not None else "📍", layout="wide")
st.markdown("<h1 style='font-size:40px;margin:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Ahlan','Samin Jaya']
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

with st.container(border=True):
    base = df_visit.copy()
    base["employee_name"] = base["employee_name"].astype(str).str.strip()
    bde_list = sorted([x for x in base["employee_name"].dropna().unique().tolist() if x])
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
        all_areas = sorted(avail_df_merge["cluster"].dropna().astype(str).unique().tolist())
        area_pick = st.multiselect("Choose Area", ["All"] + all_areas, default=["All"])
        if "All" in area_pick or not area_pick:
            area_pick = all_areas
        if name in sales_jabo or name == "All":
            jabo = list(avail_df_merge[(avail_df_merge.city.isin(jabodetabek))]['city'].dropna().unique())
            jabo = sorted(jabo)
            city_pick = st.multiselect("Choose City", ["All"] + jabo, default=["All"])
            if "All" in city_pick or not city_pick:
                city_pick = jabo
        else:
            regional = list(avail_df_merge[(~avail_df_merge.city.isin(jabodetabek))]['city'].dropna().unique())
            regional = sorted(regional)
            city_pick = st.multiselect("Choose City", ["All"] + regional, default=["All"])
            if "All" in city_pick or not city_pick:
                city_pick = regional
    if name == "All":
        brand_choose = list(avail_df_merge[(avail_df_merge.city.isin(city_pick))&(avail_df_merge.cluster.astype(str).isin(area_pick))&(avail_df_merge.tag.isin(penetrated))&(avail_df_merge.availability.isin(potential))]['brand'].dropna().unique())
    else:
        brand_choose = list(avail_df_merge[(avail_df_merge.sales_name == name)&(avail_df_merge.city.isin(city_pick))&(avail_df_merge.cluster.astype(str).isin(area_pick))&(avail_df_merge.tag.isin(penetrated))&(avail_df_merge.availability.isin(potential))]['brand'].dropna().unique())
    brand_choose = sorted(brand_choose)
    brand = st.multiselect("Choose Brand", ["All"] + brand_choose, default=["All"])
    if "All" in brand or not brand:
        brand = brand_choose
    button = st.button("Submit")

def build_pick(name, radius):
    lst = []
    if name == "All":
        names = sum_df["sales_name"].dropna().unique().tolist()
    else:
        names = [name]
    for nm in names:
        centers_cnt = len(sum_df[sum_df.sales_name == nm]["cluster"].unique())
        for i in range(centers_cnt):
            col = f"dist_center_{i}"
            if col in avail_df_merge.columns:
                tmp = avail_df_merge[(avail_df_merge[col].notna()) & (avail_df_merge[col] <= radius)]
                if name != "All":
                    tmp = tmp[tmp.sales_name == nm]
                if not tmp.empty:
                    t = tmp.copy()
                    t["cluster_labels"] = i
                    lst.append(t)
    if not lst:
        return pd.DataFrame()
    out = pd.concat(lst, ignore_index=True)
    drop_cols = [c for c in out.columns if c.startswith("dist_center_")]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    out["joined_dse"] = out["joined_dse"].fillna(0)
    out["active_dse"] = out["active_dse"].fillna(0)
    out["tag"] = np.where((out["joined_dse"].fillna(0)==0)&(out["active_dse"].fillna(0)==0),"Not Penetrated",out["tag"])
    out["nearest_end_date"] = out["nearest_end_date"].astype(str)
    out["nearest_end_date"] = np.where(out["nearest_end_date"]=="NaT","No Package Found",out["nearest_end_date"])
    return out

if not button:
    st.info("Set filters and click Submit.")
else:
    with st.spinner("Computing view..."):
        pick_avail = build_pick(name, radius)
        if pick_avail.empty:
            st.info(f"No dealers found within {radius} km.")
        else:
            if name in sales_jabo or name == "All":
                pick_avail = pick_avail[pick_avail.cluster.astype(str) == "Jabodetabek"]
            else:
                pick_avail = pick_avail[pick_avail.cluster.astype(str) != "Jabodetabek"]
            if penetrated and potential and city_pick:
                dealer_rec = pick_avail[(pick_avail.city.isin(city_pick))&(pick_avail.availability.isin(potential))&(pick_avail.tag.isin(penetrated))&(pick_avail.brand.isin(brand))&(pick_avail.cluster.astype(str).isin(area_pick))].copy()
            else:
                dealer_rec = pick_avail.copy()
            if dealer_rec.empty:
                st.info(f"No dealers match the filters within {radius} km.")
            else:
                if name == "All":
                    centers = clust_df.copy()
                else:
                    centers = clust_df[clust_df.sales_name == name].copy()
                cnt = sum_df if name == "All" else sum_df[sum_df.sales_name == name]
                count_visit_cluster = cnt[["cluster","sales_name"]].groupby("cluster").count().reset_index().rename(columns={"sales_name":"count_visit"})
                centers = pd.merge(centers, count_visit_cluster, on="cluster", how="left")
                centers["count_visit"] = centers["count_visit"].fillna(0)
                total_visits = max(centers["count_visit"].sum(), 1)
                centers["size"] = centers["count_visit"]/total_visits*9000
                centers["area_tag"] = centers["cluster"].astype(int) + 1
                centers["word"] = "Area " + centers["area_tag"].astype(str) + "\nCount Visit: " + centers["count_visit"].astype(int).astype(str)
                centers["word_pick"] = "Area " + centers["area_tag"].astype(str)
                dealer_rec["area_tag"] = dealer_rec["cluster_labels"].fillna(0).astype(int) + 1
                dealer_rec["area_tag_word"] = "Area " + dealer_rec["area_tag"].astype(str)
                m_center_lon = float(centers["longitude"].mean()) if not centers.empty else float(dealer_rec["longitude"].mean())
                m_center_lat = float(centers["latitude"].mean()) if not centers.empty else float(dealer_rec["latitude"].mean())
                st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
                layer_text = pdk.Layer(
                    "TextLayer",
                    data=centers,
                    get_position="[longitude,latitude]",
                    get_text="word",
                    get_size=12,
                    get_color=[0, 100, 0],
                    get_angle=0,
                    get_text_anchor=String("middle"),
                    get_alignment_baseline=String("center"),
                )
                layer_dealer = pdk.Layer(
                    "ScatterplotLayer",
                    data=dealer_rec.assign(_color=np.where(dealer_rec["tag"].eq("Active"), 1, np.where(dealer_rec["tag"].eq("Not Active"), 2, 3))),
                    get_position="[longitude,latitude]",
                    get_radius=200,
                    get_fill_color=["case", ["==", "_color", 1], [21,255,87,200], ["==","_color",2],[255,99,132,200],[255,206,86,200]],
                    id="dealer",
                    pickable=True,
                    auto_highlight=True,
                )
                layer_center = pdk.Layer(
                    "ScatterplotLayer",
                    data=centers,
                    get_position="[longitude,latitude]",
                    get_radius="size",
                    get_fill_color=[200,30,0,90],
                )
                st.pydeck_chart(
                    pdk.Deck(
                        map_style=None,
                        initial_view_state=pdk.ViewState(longitude=m_center_lon, latitude=m_center_lat, zoom=10, pitch=50),
                        tooltip={"text":"Dealer: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
                        layers=[layer_text, layer_dealer, layer_center],
                    )
                )
                d_now = pd.Timestamp.utcnow().date()
                v30 = df_visit[df_visit["date"] >= d_now - pd.Timedelta(days=28)]
                if name != "All":
                    v30 = v30[v30["employee_name"] == name]
                v30_cnt = v30.groupby("client_name", dropna=False).size().rename("visits_28d").reset_index()
                dsum = dealer_rec.merge(v30_cnt, how="left", left_on="name", right_on="client_name")
                visits_avg_week = float(dsum["visits_28d"].fillna(0).sum())/4.0
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Dealers", int(dealer_rec["id_dealer_outlet"].nunique()))
                with m2:
                    st.metric("Active Dealers", int((dealer_rec["active_dse"].fillna(0) > 0).sum()))
                with m3:
                    st.metric("Active DSE", int(dealer_rec["active_dse"].fillna(0).sum()))
                with m4:
                    st.metric("Avg Weekly Visits", round(visits_avg_week, 2))
                def some_output(area):
                    df_output = dealer_rec[dealer_rec.area_tag_word == area][["brand","name","city","tag","joined_dse","active_dse","nearest_end_date","availability"]].copy()
                    st.markdown(f"<h3 style='font-size:18px;margin:8px 0'>There are {len(df_output)} dealers in the radius {radius} km</h3>", unsafe_allow_html=True)
                    if not df_output.empty:
                        bar_src = df_output[["brand","tag","city"]].groupby(["brand","tag"]).count().reset_index().rename(columns={"tag":"seg","city":"Count Dealers","brand":"Brand"}).sort_values("seg")
                        fig = px.bar(bar_src, x="Brand", y="Count Dealers", hover_data=["Brand","Count Dealers"], color="seg", color_discrete_map={"Not Penetrated":"#83c9ff","Not Active":"#ffabab","Active":"#ff2b2b"})
                        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        pot_df_output = df_output[["availability","brand","city"]].groupby(["availability","brand"]).count().reset_index()
                        pot_df_output = pot_df_output.rename(columns={"availability":"Availability","brand":"Brand","city":"Total Dealers"})
                        fig1 = None
                        if not pot_df_output.empty:
                            fig1 = px.sunburst(pot_df_output, path=["Availability","Brand"], values="Total Dealers", color="Availability", color_discrete_map={"Potential":"#83c9ff","Low Generation":"#ffabab","Deficit":"#ff2b2b"})
                    else:
                        fig = None
                        fig1 = None
                    col1, col2 = st.columns([2,1])
                    with col1:
                        if fig is not None:
                            st.markdown("<h4 style='font-size:16px;margin:6px 0'>Dealer Penetration</h4>", unsafe_allow_html=True)
                            st.plotly_chart(fig, key=f"bar_{area}")
                    with col2:
                        if fig1 is not None:
                            st.markdown("<h4 style='font-size:16px;margin:6px 0'>Potential Dealer</h4>", unsafe_allow_html=True)
                            st.plotly_chart(fig1, key=f"sun_{area}")
                    if not df_output.empty:
                        df_show = df_output.rename(columns={"brand":"Brand","name":"Dealer Name","city":"City","tag":"Activity","joined_dse":"Total Joined DSE","active_dse":"Total Active DSE","nearest_end_date":"Nearest Package End Date","availability":"Availability"}).drop_duplicates(subset=["Dealer Name","City","Brand"])
                        st.markdown("<h3 style='font-size:18px;margin:8px 0'>Dealers Details</h3>", unsafe_allow_html=True)
                        st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=360)
                st.markdown("<h2 style='font-size:24px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
                tabs = centers["word_pick"].dropna().unique().tolist()
                tabs = sorted(tabs) if tabs else ["Area 1"]
                for tab, area_label in zip(st.tabs(tabs), tabs):
                    with tab:
                        some_output(area_label)
