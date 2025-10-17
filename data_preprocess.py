# main.py
import streamlit as st
from PIL import Image
import plotly.express as px
import pydeck as pdk
from pydeck.types import String
import pandas as pd
import numpy as np
from data_preprocess import compute_all

st.set_page_config(page_title="Dealer Penetration Dashboard", page_icon="assets/favicon.png", layout="wide")
img = None
try:
    img = Image.open("assets/favicon.png")
except:
    img = None

computed = compute_all()
sum_df = computed.get("sum_df", pd.DataFrame())
clust_df = computed.get("clust_df", pd.DataFrame())
avail_df_merge = computed.get("avail_df_merge", pd.DataFrame())
df_visit = computed.get("df_visits", pd.DataFrame())

sales_jabo = sorted(["A. Sofyan","Nova Handoyo","Heriyanto","Aditya rifat","Riski Amrullah Zulkarnain","Rudy Setya Wibowo","Muhammad Achlan","Samin Jaya"])
jabodetabek_list = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

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
        # city choices from avail_df_merge
        cities = avail_df_merge.get("city", pd.Series([], dtype=str)).dropna().unique().tolist() if not avail_df_merge.empty else []
        cities = sorted(cities)
        city_pick = st.multiselect("Choose City", ["All"] + cities, default=["All"])
        if "All" in city_pick:
            city_pick = cities

    brand_candidates = avail_df_merge.get("brand", pd.Series([], dtype=str)).dropna().unique().tolist() if not avail_df_merge.empty else []
    brand_candidates = sorted(brand_candidates)
    brand = st.multiselect("Choose Brand", ["All"] + brand_candidates, default=["All"])
    if "All" in brand:
        brand = brand_candidates

    button = st.button("Submit")

if not button:
    st.info("Set filters and click Submit.")
else:
    # build selection
    # gather cluster indices to compute dist_center_N columns
    pick_avail_lst = []
    # find candidate dist columns
    dist_cols = [c for c in (avail_df_merge.columns if not avail_df_merge.empty else []) if c.startswith("dist_center_")]
    # for each dist column use filter radius
    if dist_cols:
        for col in dist_cols:
            temp = avail_df_merge[(avail_df_merge[col].notna()) & (pd.to_numeric(avail_df_merge[col], errors='coerce') <= radius)]
            if name != "All":
                temp = temp[temp.get("sales_name", "") == name]
            if not temp.empty:
                temp = temp.copy()
                temp['cluster_labels'] = int(col.replace("dist_center_",""))
                pick_avail_lst.append(temp)
    else:
        # fallback: use entire avail_df_merge filtered by sales_name if radius not applicable
        tmp = avail_df_merge.copy() if not avail_df_merge.empty else pd.DataFrame()
        if not tmp.empty and name != "All":
            tmp = tmp[tmp.get("sales_name","") == name]
        if not tmp.empty:
            tmp['cluster_labels'] = tmp.get('cluster', 0).fillna(0).astype(int)
            pick_avail_lst.append(tmp)

    if not pick_avail_lst:
        st.info(f"No dealers found within {radius} km.")
    else:
        pick_avail = pd.concat(pick_avail_lst, ignore_index=True)
        # filter by city/availability/tag/brand (only if those columns exist)
        if city_pick:
            if 'city' in pick_avail.columns and len(city_pick)>0:
                pick_avail = pick_avail[pick_avail.city.isin(city_pick)]
        if potential and 'availability' in pick_avail.columns:
            pick_avail = pick_avail[pick_avail.availability.isin(potential)]
        if penetrated and 'tag' in pick_avail.columns:
            pick_avail = pick_avail[pick_avail.tag.isin(penetrated)]
        if brand and 'brand' in pick_avail.columns:
            pick_avail = pick_avail[pick_avail.brand.isin(brand)]
        # fill join/active counts
        pick_avail['joined_dse'] = pick_avail.get('joined_dse', pd.Series(0)).fillna(0).astype(int)
        pick_avail['active_dse'] = pick_avail.get('active_dse', pd.Series(0)).fillna(0).astype(int)
        pick_avail['tag'] = np.where((pick_avail.joined_dse==0) & (pick_avail.active_dse==0), "Not Penetrated", pick_avail.get('tag', "Not Penetrated"))
        pick_avail['nearest_end_date'] = pd.to_datetime(pick_avail.get('nearest_end_date', pd.Series(pd.NaT)), errors="coerce").dt.strftime("%Y-%m-%d")
        # de-dupe dealers
        if 'id_dealer_outlet' in pick_avail.columns:
            dealer_rec = pick_avail.drop_duplicates(subset=['id_dealer_outlet']).reset_index(drop=True)
        else:
            dealer_rec = pick_avail.drop_duplicates().reset_index(drop=True)
        if dealer_rec.empty:
            st.info("No dealers match the filters.")
        else:
            # clustering centers for map view
            centers = clust_df.copy() if clust_df is not None else pd.DataFrame()
            if name != "All" and not centers.empty:
                centers = centers[centers.sales_name == name]
            # compute map center
            center_lon = float(centers.longitude.mean()) if 'longitude' in centers.columns and not centers.longitude.isna().all() else float(dealer_rec.longitude.mean())
            center_lat = float(centers.latitude.mean()) if 'latitude' in centers.columns and not centers.latitude.isna().all() else float(dealer_rec.latitude.mean())
            # color map
            def _col(r):
                tag = r.get('tag', 'Not Penetrated')
                if tag == "Not Penetrated":
                    return [131,201,255,200]
                if tag == "Not Active":
                    return [255,171,171,200]
                if tag == "Active":
                    return [255,43,43,200]
                return [200,200,200,200]
            dealer_rec = dealer_rec.copy()
            dealer_rec['color'] = dealer_rec.apply(_col, axis=1).tolist()
            dealer_records = dealer_rec.to_dict(orient='records')
            center_records = centers.to_dict(orient='records') if not centers.empty else []
            # Render map
            st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
            deck = pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=10, pitch=50),
                tooltip={'text': "Brand: {brand}\nPenetration: {tag}"},
                layers=[
                    pdk.Layer("TextLayer", data=center_records, get_position="[longitude,latitude]", get_text="cluster", get_size=12, get_color=[0,100,0], get_angle=0, get_text_anchor=String("middle"), get_alignment_baseline=String("center")),
                    pdk.Layer("ScatterplotLayer", data=dealer_records, get_position="[longitude,latitude]", get_radius=200, get_fill_color="color", pickable=True, auto_highlight=True),
                ]
            )
            st.pydeck_chart(deck)
            # For each area tab show charts and table
            def some_output(df_area):
                # choose safe columns intersection
                desired = ['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']
                cols = [c for c in desired if c in df_area.columns]
                df_out = df_area[cols].copy()
                st.markdown(f"### There are {len(df_out)} dealers in the radius {radius} km")
                if len(df_out)>0:
                    # bar: brand x count colored by activity if tag present
                    bar_cols = ['brand']
                    if 'tag' in df_out.columns:
                        bar_src = df_out.groupby(['brand','tag']).size().reset_index(name='Count Dealers')
                        fig = px.bar(bar_src, x='brand', y='Count Dealers', color='tag', labels={'brand':'Brand','tag':'Activity'})
                    else:
                        bar_src = df_out.groupby(['brand']).size().reset_index(name='Count Dealers')
                        fig = px.bar(bar_src, x='brand', y='Count Dealers', labels={'brand':'Brand'})
                    fig.update_layout(legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.markdown("#### Dealer Penetration")
                    st.plotly_chart(fig, use_container_width=True)
                    # potential sunburst
                    if 'availability' in df_out.columns and 'brand' in df_out.columns:
                        pot = df_out.groupby(['availability','brand']).size().reset_index(name='Total Dealers')
                        if not pot.empty:
                            fig1 = px.sunburst(pot, path=['availability','brand'], values='Total Dealers', color='availability')
                            st.markdown("#### Potential Dealer")
                            st.plotly_chart(fig1, use_container_width=True)
                # show table
                if not df_out.empty:
                    rename_map = {}
                    if 'name' in df_out.columns:
                        rename_map['name'] = 'Dealer Name'
                    if 'brand' in df_out.columns:
                        rename_map['brand'] = 'Brand'
                    if 'city' in df_out.columns:
                        rename_map['city'] = 'City'
                    if 'joined_dse' in df_out.columns:
                        rename_map['joined_dse'] = 'Total Joined DSE'
                    if 'active_dse' in df_out.columns:
                        rename_map['active_dse'] = 'Total Active DSE'
                    if 'nearest_end_date' in df_out.columns:
                        rename_map['nearest_end_date'] = 'Nearest Package End Date'
                    if 'availability' in df_out.columns:
                        rename_map['availability'] = 'Availability'
                    df_shown = df_out.rename(columns=rename_map)
                    df_shown = df_shown.drop_duplicates()
                    st.markdown("### Dealers Details")
                    st.dataframe(df_shown.reset_index(drop=True))
            # prepare tabs (areas). Use clusters if available otherwise 'All'
            if centers:
                tab_labels = sorted(list({f"Area {int(c['cluster'])+1}" for c in center_records}))
                for tab, lab in zip(st.tabs(tab_labels), tab_labels):
                    with tab:
                        if 'cluster' in dealer_rec.columns:
                            area_idx = int(lab.replace("Area ","")) - 1
                            df_area = dealer_rec[dealer_rec.get('cluster_labels', dealer_rec.get('cluster', -999)) == area_idx]
                        else:
                            df_area = dealer_rec
                        some_output(df_area)
            else:
                with st.container():
                    some_output(dealer_rec)
