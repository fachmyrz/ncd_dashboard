import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from datetime import datetime, timedelta
import math

def haversine_km_array(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return 6371.0088 * c

def clean_latlon_value(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    s = s.replace('`','').replace('’','').replace('“','').replace('”','').replace('"','').replace("'",'').strip()
    s = s.replace(',','.')
    s = s.strip()
    try:
        return float(s)
    except:
        import re
        m = re.search(r'(-?\d+\.\d+)', s)
        if m:
            try:
                return float(m.group(1))
            except:
                return np.nan
        return np.nan

def normalize_colnames(df):
    out = {}
    for c in df.columns:
        k = c.strip().lower().replace(' ', '_')
        out[c] = k
    df = df.rename(columns=out)
    return df

def clean_dealers(df_dealer):
    df = df_dealer.copy()
    df.columns = df.columns.astype(str)
    df = df.rename(columns=lambda x: x.strip())
    if 'business_type' in df.columns:
        df['business_type'] = df['business_type'].astype(str).str.strip()
    df = df[df.get('business_type', '').astype(str).str.lower().str.contains('car', na=False)]
    if 'latitude' in df.columns:
        df['latitude'] = df['latitude'].apply(clean_latlon_value)
    if 'longitude' in df.columns:
        df['longitude'] = df['longitude'].apply(clean_latlon_value)
    df = df.dropna(subset=['latitude','longitude']).reset_index(drop=True)
    if 'id_dealer_outlet' in df.columns:
        df['id_dealer_outlet'] = pd.to_numeric(df['id_dealer_outlet'], errors='coerce').astype('Int64')
    df['brand'] = df.get('brand','').astype(str).str.strip()
    df['city'] = df.get('city','').astype(str).str.strip()
    df['name'] = df.get('name','').astype(str).str.strip()
    return df

def parse_visit_latlong(val):
    if pd.isna(val):
        return (np.nan, np.nan)
    s = str(val).strip()
    s = s.replace('`','').replace('"','').replace("'",'').strip()
    parts = [p.strip() for p in s.replace(';',',').split(',') if p.strip()!='']
    if len(parts) >= 2:
        lat = clean_latlon_value(parts[0])
        lon = clean_latlon_value(parts[1])
        return (lat, lon)
    else:
        import re
        m = re.findall(r'(-?\d+\.\d+)', s)
        if len(m) >= 2:
            return (clean_latlon_value(m[0]), clean_latlon_value(m[1]))
    return (np.nan, np.nan)

def clean_visits(df_visits_raw):
    df = df_visits_raw.copy()
    df.columns = df.columns.astype(str)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if 'employee' in cl and ('name' in cl or 'karyawan' in cl):
            col_map[c] = 'employee_name'
        if 'client' in cl or 'nama_klien' in cl or 'nama_klien' in cl:
            col_map[c] = 'client_name'
        if 'tanggal datang' in cl or 'date time start' in cl or 'date_time_start' in cl:
            col_map[c] = 'date_time_start'
        if 'date time end' in cl or 'tanggal selesai' in cl or 'date_time_end' in cl:
            col_map[c] = 'date_time_end'
        if 'latitude' in cl and 'longitude' in cl and 'datang' in cl:
            col_map[c] = 'latlong'
        if 'latitude' in cl and ('start' in cl or 'datang' in cl) and 'longitude' not in cl:
            col_map[c] = 'lat'
        if 'longitude' in cl and ('start' in cl or 'datang' in cl) and 'latitude' not in cl:
            col_map[c] = 'lon'
        if 'nomor_induk' in cl or 'nomor' in cl and 'karyawan' in cl:
            col_map[c] = 'nik'
        if 'divisi' in cl:
            col_map[c] = 'divisi'
    df = df.rename(columns=col_map)
    if 'latlong' in df.columns:
        parsed = df['latlong'].apply(parse_visit_latlong)
        df['lat'] = parsed.apply(lambda t: t[0])
        df['long'] = parsed.apply(lambda t: t[1])
    else:
        if 'lat' in df.columns:
            df['lat'] = df['lat'].apply(clean_latlon_value)
        if 'long' in df.columns:
            df['long'] = df['long'].apply(clean_latlon_value)
    df['date_time_orig'] = df.get('date_time_start', pd.NaT)
    if 'date_time_start' in df.columns:
        def parse_dt(x):
            if pd.isna(x):
                return pd.NaT
            s = str(x)
            if '@' in s:
                parts = s.split('@')
                s = parts[0].strip() + ' ' + parts[1].strip()
            try:
                return pd.to_datetime(s, errors='coerce')
            except:
                try:
                    return pd.to_datetime(s, dayfirst=True, errors='coerce')
                except:
                    return pd.NaT
        df['visit_datetime'] = df['date_time_start'].apply(parse_dt)
    else:
        df['visit_datetime'] = pd.NaT
    df['date'] = pd.to_datetime(df['visit_datetime'].dt.date, errors='coerce')
    df['time'] = df['visit_datetime'].dt.time
    df['duration_min'] = np.nan
    if 'date_time_end' in df.columns and df['date_time_end'].notna().any():
        def parse_dt_end(x):
            if pd.isna(x):
                return pd.NaT
            s = str(x)
            if '@' in s:
                parts = s.split('@')
                s = parts[0].strip() + ' ' + parts[1].strip()
            try:
                return pd.to_datetime(s, errors='coerce')
            except:
                try:
                    return pd.to_datetime(s, dayfirst=True, errors='coerce')
                except:
                    return pd.NaT
        df['visit_end'] = df['date_time_end'].apply(parse_dt_end)
        df['duration_min'] = (df['visit_end'] - df['visit_datetime']).dt.total_seconds().div(60)
    df['lat'] = df['lat'].apply(clean_latlon_value)
    df['long'] = df['long'].apply(clean_latlon_value)
    df = df[~df['employee_name'].isna()].reset_index(drop=True)
    return df

def assign_visits_to_dealers(visits, dealers, max_km=1.0):
    v = visits.copy()
    d = dealers.copy()
    d = d[['id_dealer_outlet','name','latitude','longitude']].dropna().reset_index(drop=True)
    d_lat = d['latitude'].to_numpy(dtype=float)
    d_lon = d['longitude'].to_numpy(dtype=float)
    def match_row(row):
        lat = row.get('lat', np.nan)
        lon = row.get('long', np.nan)
        cname = row.get('client_name', '')
        if pd.notna(cname):
            cand = d[d['name'].astype(str).str.strip().str.lower() == str(cname).strip().lower()]
            if not cand.empty:
                return cand.iloc[0]['name']
        if pd.notna(lat) and pd.notna(lon):
            dist_arr = haversine_km_array(float(lat), float(lon), d_lat, d_lon)
            idx = np.argmin(dist_arr)
            if dist_arr[idx] <= max_km:
                return d.iloc[int(idx)]['name']
        return np.nan
    v['matched_client'] = v.apply(match_row, axis=1)
    return v

def get_summary_data(df_visits, pick_date="2024-11-01"):
    summary_base = df_visits.copy()
    try:
        cutoff = pd.to_datetime(pick_date).date()
    except:
        cutoff = pd.to_datetime("2024-11-01").date()
    summary = summary_base[summary_base['date'] >= cutoff].copy()
    summary['lat'] = pd.to_numeric(summary['lat'], errors='coerce')
    summary['long'] = pd.to_numeric(summary['long'], errors='coerce')
    data = []
    for dt in summary['date'].dropna().unique():
        for name in summary['employee_name'].dropna().unique():
            temp = summary[(summary['employee_name']==name)&(summary['date']==dt)]
            temp = temp.sort_values('visit_datetime')
            if len(temp) > 1:
                lat = temp['lat'].to_numpy(dtype=float)
                lon = temp['long'].to_numpy(dtype=float)
                dists = []
                times = []
                for i in range(len(temp)-1):
                    a = lat[i]; b = lon[i]; c = lat[i+1]; e = lon[i+1]
                    if pd.isna(a) or pd.isna(b) or pd.isna(c) or pd.isna(e):
                        continue
                    dkm = haversine_km_array(a,b,c,e)
                    dists.append(float(dkm))
                    tdiff = (pd.to_datetime(temp.iloc[i+1]['visit_datetime']) - pd.to_datetime(temp.iloc[i]['visit_datetime'])).total_seconds()/60
                    times.append(float(tdiff))
                if len(dists)==0:
                    avg_d = 0
                    avg_t = 0
                    avg_speed = 0
                else:
                    avg_d = round(float(np.mean(dists)),2)
                    avg_t = round(float(np.mean(times)),2)
                    avg_speed = round(float(sum(dists)/sum(times)) if sum(times)>0 else 0,4)
                data.append([dt,name,len(temp),avg_d,avg_t,avg_speed])
            else:
                data.append([dt,name,len(temp),0.0,0.0,0.0])
    cols = ['date','employee_name','ctd_visit','avg_distance_km','avg_time_between_minute','avg_speed_kmpm']
    dataf = pd.DataFrame(data, columns=cols)
    dataf['month_year'] = dataf['date'].astype(str).apply(lambda x: '-'.join(x.split('-')[:2]) if isinstance(x,str) and '-' in x else str(x))
    return summary, dataf

def compute_clusters_and_distances(df_visits, df_dealers):
    clust_list = []
    avail_list = []
    sum_list = []
    for name in df_visits['employee_name'].dropna().unique():
        sv = df_visits[df_visits['employee_name']==name].dropna(subset=['lat','long']).copy()
        if sv.empty:
            continue
        n_points = len(sv)
        if n_points >= 2:
            max_k = min(8, n_points) if n_points>=4 else max(1,n_points)
            k_range = list(range(2, max(4, max_k+1)))
            wcss = []
            coords = sv[['lat','long']].to_numpy()
            for k in k_range:
                try:
                    km = KMeans(n_clusters=k, random_state=42).fit(coords)
                    wcss.append(km.inertia_)
                except:
                    wcss.append(np.nan)
            try:
                kneed = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
                n_cluster = int(kneed.elbow) if kneed.elbow is not None else min(4, max(1,n_points))
            except:
                n_cluster = min(4, max(1,n_points))
            kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(coords)
            centers = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude','longitude'])
            centers['sales_name'] = name
            centers['cluster'] = centers.index
            for i in range(len(kmeans.cluster_centers_)):
                latc = kmeans.cluster_centers_[i][0]
                lonc = kmeans.cluster_centers_[i][1]
            sv['cluster'] = kmeans.labels_
            sum_list.append(sv.assign(sales_name=name))
        else:
            centers = pd.DataFrame([[sv.iloc[0]['lat'], sv.iloc[0]['long']]], columns=['latitude','longitude'])
            centers['sales_name'] = name
            centers['cluster'] = range(len(centers))
            sv['cluster'] = 0
            sum_list.append(sv.assign(sales_name=name))
        clust_list.append(centers)
    if clust_list:
        clust_df = pd.concat(clust_list, ignore_index=True)
    else:
        clust_df = pd.DataFrame(columns=['latitude','longitude','sales_name','cluster'])
    if sum_list:
        sum_df = pd.concat(sum_list, ignore_index=True)
    else:
        sum_df = pd.DataFrame(columns=['date','employee_name','lat','long','visit_datetime','cluster','sales_name'])
    avail_df = df_dealers.copy()
    for i,row in clust_df.reset_index().iterrows():
        idx = int(row['cluster'])
        col = f'dist_center_{idx}'
        avail_df[col] = haversine_km_array(row['latitude'], row['longitude'], avail_df['latitude'].to_numpy(dtype=float), avail_df['longitude'].to_numpy(dtype=float))
    return sum_df, clust_df, avail_df

def compute_running_orders_and_packages(running_order_df):
    ro = running_order_df.copy()
    ro = ro.rename(columns=lambda x: x.strip())
    if 'IsActive' in ro.columns:
        ro_active = ro[ro['IsActive'].astype(str).str.strip()=='1'].copy()
    else:
        ro_active = ro[ro.get('IsActive', pd.Series([])).astype(str).str.strip()=='1'].copy()
    if 'End Date' in ro_active.columns:
        ro_active['End Date'] = pd.to_datetime(ro_active['End Date'], errors='coerce')
    ro_active['Dealer Id'] = pd.to_numeric(ro_active.get('Dealer Id', ro_active.get('Dealer Id', pd.Series([]))), errors='coerce').astype('Int64')
    if 'Dealer Id' in ro_active.columns:
        ao_group = ro_active.groupby(['Dealer Id','Dealer Name']).agg({'End Date':'min'}).reset_index()
        ao_group = ao_group.rename(columns={'Dealer Id':'id_dealer_outlet','Dealer Name':'dealer_name','End Date':'nearest_end_date'})
    else:
        ao_group = pd.DataFrame(columns=['id_dealer_outlet','dealer_name','nearest_end_date'])
    ro2 = ro.copy()
    ro2['Dealer Id'] = pd.to_numeric(ro2.get('Dealer Id', pd.Series([])), errors='coerce').astype('Int64')
    ro2['active_dse'] = pd.to_numeric(ro2.get('IsActive', 0), errors='coerce').fillna(0).astype('Int64')
    ro2['joined_dse'] = 1
    grouped = ro2.groupby(['Dealer Id']).agg({'joined_dse':'sum','active_dse':'sum'}).reset_index().rename(columns={'Dealer Id':'id_dealer_outlet'})
    grouped['id_dealer_outlet'] = grouped['id_dealer_outlet'].astype('Int64')
    merged = pd.merge(grouped, ao_group, how='left', on='id_dealer_outlet')
    return merged

def build_avail_merge(avail_df, run_order_group, location_detail_df, need_cluster_df):
    avail = avail_df.copy()
    dist_cols = [c for c in avail.columns if c.startswith('dist_center_')]
    if dist_cols:
        for c in dist_cols:
            avail[c] = pd.to_numeric(avail[c], errors='coerce')
    run_order_group = run_order_group.copy()
    run_order_group['id_dealer_outlet'] = run_order_group['id_dealer_outlet'].astype('Int64', errors='ignore')
    if 'id_dealer_outlet' in avail.columns:
        avail['id_dealer_outlet'] = pd.to_numeric(avail['id_dealer_outlet'], errors='coerce').astype('Int64')
    ld = location_detail_df.copy()
    ld = ld.rename(columns=lambda x: x.strip())
    if 'City' in ld.columns and 'Cluster' in ld.columns:
        ld = ld.rename(columns={'City':'city','Cluster':'cluster'})
    else:
        ld = ld.rename(columns=lambda x: x.lower())
    avail_merge = pd.merge(avail, run_order_group, how='left', on='id_dealer_outlet')
    if 'city' in avail_merge.columns and 'city' in ld.columns:
        avail_merge = avail_merge.merge(ld[['city','cluster']], how='left', on='city')
    if need_cluster_df is not None and not need_cluster_df.empty:
        nc = need_cluster_df.copy()
        nc = nc.rename(columns=lambda x: x.strip())
        if 'Cluster' in nc.columns:
            nc = nc.rename(columns={'Cluster':'cluster','Brand':'brand','Daily_Gen':'daily_gen','Daily_Need':'daily_need','Delta':'delta','Tag':'availability','Category':'Category'})
        nc['brand'] = nc.get('brand','').astype(str).str.strip()
        nc['cluster'] = nc.get('cluster','').astype(str).str.strip()
        nc_car = nc[nc.get('Category','').astype(str).str.lower()=='car'].copy()
        avail_merge = pd.merge(avail_merge, nc_car[['cluster','brand','daily_gen','daily_need','delta','availability']], how='left', on=['brand','cluster'])
    avail_merge['nearest_end_date'] = pd.to_datetime(avail_merge.get('nearest_end_date', pd.NaT), errors='coerce')
    avail_merge['tag'] = np.where(avail_merge['nearest_end_date'].isna(), 'Not Active', 'Active')
    avail_merge['joined_dse'] = pd.to_numeric(avail_merge.get('joined_dse', 0), errors='coerce').fillna(0).astype(int)
    avail_merge['active_dse'] = pd.to_numeric(avail_merge.get('active_dse', 0), errors='coerce').fillna(0).astype(int)
    avail_merge['tag'] = np.where((avail_merge.joined_dse==0)&(avail_merge.active_dse==0),'Not Penetrated',avail_merge['tag'])
    return avail_merge

def compute_all(sheets):
    dealers = sheets.get('dealers', pd.DataFrame())
    visits = sheets.get('visits', pd.DataFrame())
    location = sheets.get('location', pd.DataFrame())
    need_cluster = sheets.get('need_cluster', pd.DataFrame())
    running_order = sheets.get('running_order', pd.DataFrame())
    dealers = clean_dealers(dealers)
    visits = clean_visits(visits)
    visits = visits[~visits.get('nik', pd.Series('', dtype=str)).astype(str).str.contains('deleted-', na=False)]
    visits = visits[~visits.get('divisi', pd.Series('', dtype=str)).astype(str).str.contains('trainer', na=False)]
    visits = assign_visits_to_dealers(visits, dealers, max_km=1.0)
    summary, data_sum = get_summary_data(visits)
    sum_df, clust_df, avail_df = compute_clusters_and_distances(visits, dealers)
    run_group = compute_running_orders_and_packages(running_order)
    avail_df_merge = build_avail_merge(avail_df, run_group, location, need_cluster)
    return dict(sum_df=sum_df, clust_df=clust_df, avail_df_merge=avail_df_merge, df_visits=visits, summary=summary, data_sum=data_sum)
