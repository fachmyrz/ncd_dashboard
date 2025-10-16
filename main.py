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

sales_jabo = ['A. Sofyan','Nova Handoyo','Heriyanto','Aditya rifat','Riski Amrullah Zulkarnain','Rudy Setya Wibowo','Muhammad Ahlan','Samin Jaya']
jabodetabek = ['Bekasi','Bogor','Depok','Jakarta Barat','Jakarta Pusat','Jakarta Selatan','Jakarta Timur','Jakarta Utara','Tangerang','Tangerang Selatan','Cibitung','Tambun','Cikarang','Karawaci','Alam Sutera','Cileungsi','Sentul','Cibubur','Bintaro']

st.markdown("<h1 style='font-size:40px;margin-top:0'>Filter for Recommendation</h1>", unsafe_allow_html=True)

with st.container(border=True):
  bde_base = df_visit.copy()
  nik_mask = ~bde_base.get("Nomor Induk Karyawan", pd.Series([],dtype=str)).astype(str).str.contains("deleted-", case=False, na=False)
  div_mask = ~bde_base.get("Divisi", pd.Series([],dtype=str)).astype(str).str.contains("Trainer", case=False, na=False)
  bde_opts = sorted(bde_base.loc[nik_mask & div_mask, "employee_name"].dropna().unique().tolist())
  bde_opts = ["All"] + bde_opts
  name = st.selectbox("BDE Name ", bde_opts, index=0)

  all_areas = sorted(avail_df_merge['cluster'].dropna().astype(str).unique().tolist())
  area_mult = st.multiselect("Area", ["All"] + all_areas, default=["All"])
  if "All" in area_mult:
      area_selected = all_areas
  else:
      area_selected = area_mult

  cols1 = st.columns(2)
  with cols1[0]:
    penetrated = st.multiselect("Dealer Activity",['All','Not Active','Not Penetrated','Active'], default=['All'])
    if "All" in penetrated:
      penetrated = ['Not Active','Not Penetrated','Active']
    radius = st.slider("Choose Radius",0,50,15)
  with cols1[1]:
    potential =  st.multiselect("Dealer Availability",['All','Potential','Low Generation','Deficit'], default=['All'])
    if "All" in potential:
      potential = ['Potential','Low Generation','Deficit']
    city_pool = avail_df_merge[avail_df_merge['cluster'].isin(area_selected)]['city'].dropna().astype(str).unique().tolist()
    city_pick = st.multiselect("Choose City", ["All"] + sorted(city_pool), default=["All"])
    if "All" in city_pick:
        city_pick = city_pool

  brand_pool = avail_df_merge[
      (avail_df_merge['cluster'].isin(area_selected)) &
      (avail_df_merge['city'].isin(city_pick)) &
      (avail_df_merge['tag'].isin(penetrated)) &
      (avail_df_merge['availability'].isin(potential))
  ]['brand'].dropna().astype(str).unique().tolist()
  brand = st.multiselect("Choose Brand", ["All"] + sorted(brand_pool), default=["All"])
  if "All" in brand:
    brand = brand_pool

  button = st.button("Submit")

if button and name != "" and penetrated != "" and potential != "" and city_pick != "":
  if name == "All":
      names_scope = sum_df['sales_name'].dropna().unique().tolist()
  else:
      names_scope = [name]

  pick_avail_lst = []
  for nm in names_scope:
    clusters = sum_df[sum_df.sales_name == nm]['cluster'].dropna().unique().tolist()
    for i in clusters:
      col = f'dist_center_{int(i)}'
      if col in avail_df_merge.columns:
        temp_pick = avail_df_merge[
            (avail_df_merge[col].notna())&
            (avail_df_merge[col] <= radius)&
            (avail_df_merge.sales_name == nm)
        ].copy()
        if not temp_pick.empty:
          temp_pick['cluster_labels'] = int(i)
          pick_avail_lst.append(temp_pick)

  if not pick_avail_lst:
    st.info(f"No dealers found within {radius} km.")
  else:
    pick_avail = pd.concat(pick_avail_lst, ignore_index=True)
    drop_cols = [c for c in pick_avail.columns if c.startswith('dist_center_')]
    if drop_cols:
      pick_avail.drop(columns=drop_cols, inplace=True, errors='ignore')

    pick_avail['joined_dse'] = pd.to_numeric(pick_avail['joined_dse'], errors='coerce').fillna(0)
    pick_avail['active_dse'] = pd.to_numeric(pick_avail['active_dse'], errors='coerce').fillna(0)
    pick_avail['tag'] = np.where((pick_avail.joined_dse==0)&(pick_avail.active_dse==0),"Not Penetrated",pick_avail.tag)
    pick_avail['nearest_end_date'] = pd.to_datetime(pick_avail['nearest_end_date'], errors='coerce').astype(str)
    pick_avail['nearest_end_date'] = np.where(pick_avail['nearest_end_date'] == 'NaT',"No Package Found",pick_avail['nearest_end_date'])

    pick_avail = pick_avail[pick_avail['cluster'].isin(area_selected)]

    if penetrated != "" and potential != "" and city_pick != "":
      pick_avail_filter = pick_avail[
        (pick_avail.city.astype(str).isin(city_pick))&
        (pick_avail.availability.isin(potential))&
        (pick_avail.tag.isin(penetrated))&
        (pick_avail.brand.astype(str).isin(brand))
      ].copy()
    else:
      pick_avail_filter = pick_avail.copy()

    dealer_rec = pick_avail_filter.copy()
    if dealer_rec.empty:
      st.info(f"No dealers match the filters within {radius} km.")
    else:
      dealer_rec.sort_values(['cluster_labels','delta','latitude','longitude'],ascending=False,inplace=True)

      if name == "All":
        cluster_center = clust_df[clust_df.sales_name.isin(names_scope)].copy()
      else:
        cluster_center = clust_df[clust_df.sales_name == name].copy()

      count_visit_cluster = sum_df[sum_df.sales_name.isin(names_scope)][['cluster','sales_name']].groupby('cluster').count().reset_index().rename(columns={'sales_name':'count_visit'})
      cluster_center = pd.merge(cluster_center,count_visit_cluster,on='cluster',how='left')
      cluster_center['count_visit'] = cluster_center['count_visit'].fillna(0)
      total_visits = max(cluster_center['count_visit'].sum(), 1)
      cluster_center['size'] = cluster_center['count_visit']/total_visits*9000
      cluster_center['area_tag'] = cluster_center['cluster'].astype(int) + 1
      cluster_center['word'] = "Area " + cluster_center['area_tag'].astype(str) + "\nCount Visit: " + cluster_center['count_visit'].astype(int).astype(str)
      cluster_center['word_pick'] = "Area " + cluster_center['area_tag'].astype(str)

      st.markdown("<h2 style='font-size:24px;margin:8px 0'>Penetration Map</h2>", unsafe_allow_html=True)
      st.pydeck_chart(
        pdk.Deck(
          map_style=None,
          initial_view_state=pdk.ViewState(
            longitude=float(dealer_rec['longitude'].astype(float).mean()) if dealer_rec['longitude'].notna().any() else 106.8456,
            latitude=float(dealer_rec['latitude'].astype(float).mean()) if dealer_rec['latitude'].notna().any() else -6.2088,
            zoom=10,
            pitch=50
          ),
          tooltip={'text':"Dealer Name: {name}\nBrand: {brand}\nAvailability: {availability}\nPenetration: {tag}"},
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
              get_alignment_baseline=String("center"),
              id="lbl"
            ),
            pdk.Layer(
              "ScatterplotLayer",
              data=dealer_rec,
              get_position="[longitude,latitude]",
              get_radius=200,
              get_color="[21, 255, 87, 200]",
              id="dealer",
              pickable=True,
              auto_highlight=True
            ),
            pdk.Layer(
              "ScatterplotLayer",
              data=cluster_center,
              get_position="[longitude,latitude]",
              get_radius="size",
              get_color="[200, 30, 0, 90]",
              id="centers"
            ),
          ]
        )
      )

      def some_output(area):
        df_output = dealer_rec[dealer_rec.area_tag_word == area]
        df_output = df_output[['brand','name','city','tag','joined_dse','active_dse','nearest_end_date','availability']].copy()
        st.markdown(f"### There are {len(df_output)} dealers in the radius {radius} km")
        if not df_output.empty:
          bar_src = (
            df_output[['brand','tag','city']]
            .groupby(['brand','tag']).count().reset_index()
            .rename(columns={'tag':' ','city':'Count Dealers','brand':'Brand'})
            .sort_values(' ')
          )
          fig = px.bar(
            bar_src,
            x='Brand',
            y='Count Dealers',
            hover_data=['Brand','Count Dealers'],
            color=' ',
            color_discrete_map={'Not Penetrated':"#83c9ff",'Not Active':'#ffabab','Active':"#ff2b2b"}
          )
          fig.update_layout(legend=dict(orientation='h',yanchor="bottom",y=1.02,xanchor="right",x=1))
          pot_df_output = df_output[['availability','brand','city']].groupby(['availability','brand']).count().reset_index()
          pot_df_output.rename(columns={'availability':'Availability','brand':'Brand','city':'Total Dealers'},inplace=True)
          if not pot_df_output.empty:
            fig1 = px.sunburst(
              pot_df_output,
              path=['Availability', 'Brand'],
              values='Total Dealers',
              color='Availability',
              color_discrete_map={'Potential':"#83c9ff",'Low Generation':'#ffabab','Deficit':"#ff2b2b"}
            )
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
          df_shown = df_output.rename(columns={
            'brand':'Brand',
            'name':'Dealer',
            'city':'City',
            'tag':'Activity',
            'joined_dse':'Total Joined DSE',
            'active_dse':'Total Active DSE',
            'nearest_end_date':'Nearest Package End Date',
            'availability':'Availability'
          }).drop_duplicates(subset=['Dealer'])
          st.markdown("### Dealers Details")
          st.dataframe(df_shown.reset_index(drop=True), use_container_width=True, key=f"tbl_{area}")

      dealer_rec['area_tag'] = dealer_rec['cluster_labels'].astype(int) + 1
      dealer_rec['area_tag_word'] = "Area " + dealer_rec['area_tag'].astype(str)

      st.markdown("<h2 style='font-size:24px;margin:8px 0'>Dealers Detail</h2>", unsafe_allow_html=True)
      tab_labels = cluster_center['word_pick'].dropna().unique().tolist()
      tab_labels.sort()
      for tab,area_label in zip(st.tabs(tab_labels if tab_labels else ["No Area"]), tab_labels if tab_labels else ["No Area"]):
          with tab:
              if area_label == "No Area":
                  st.info("No areas to display.")
              else:
                  some_output(area_label)
