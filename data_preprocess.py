import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
import geopy.distance
from data_load import get_sheets

pd.options.mode.copy_on_write = True

def _to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("`","").replace("’","").replace("“","").replace("”","")
    s = s.replace(" ", "").replace(",", "") if s.count(",")>1 else s.replace(" ", "")
    try:
        return float(s)
    except Exception:
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return np.nan

def _valid_latlon(lat, lon):
    try:
        if pd.isna(lat) or pd.isna(lon):
            return False
        latf = float(lat)
        lonf = float(lon)
        if not np.isfinite(latf) or not np.isfinite(lonf):
            return False
        return -90.0 <= latf <= 90.0 and -180.0 <= lonf <= 180.0
    except Exception:
        return False

def _km(lat1, lon1, lat2, lon2):
    if not _valid_latlon(lat1, lon1) or not _valid_latlon(lat2, lon2):
        return np.inf
    try:
        return geopy.distance.geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km
    except Exception:
        return np.inf

def split_latlon(col):
    if pd.isna(col):
        return np.nan, np.nan
    s = str(col).strip().replace("`","").replace("’","")
    if ";" in s:
        s = s.replace(";", ",")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) >= 2:
        return _to_float(parts[0]), _to_float(parts[1])
    return np.nan, np.nan

def _valid_mask(lat_series, lon_series):
    return pd.Series([_valid_latlon(a,b) for a,b in zip(lat_series, lon_series)], index=lat_series.index)

def clean_dealers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = {c.lower().strip(): c for c in df.columns}
    id_col = cols.get("id_dealer_outlet") or cols.get("dealer_id") or list(df.columns)[0]
    brand_col = cols.get("brand") or "brand"
    bt_col = cols.get("business_type") or "business_type"
    city_col = cols.get("city") or "city"
    name_col = cols.get("name") or cols.get("dealer_name") or "name"
    lat_col = cols.get("latitude") or "latitude"
    lon_col = cols.get("longitude") or "longitude"
    out = df.copy()
    out.rename(columns={id_col:"id_dealer_outlet",brand_col:"brand",bt_col:"business_type",city_col:"city",name_col:"name",lat_col:"latitude",lon_col:"longitude"}, inplace=True)
    if "Latitude & Longitude" in out.columns:
        lat, lon = zip(*out["Latitude & Longitude"].apply(split_latlon))
        out["latitude"] = lat
        out["longitude"] = lon
    out["latitude"] = out["latitude"].apply(_to_float)
    out["longitude"] = out["longitude"].apply(_to_float)
    out["id_dealer_outlet"] = pd.to_numeric(out["id_dealer_outlet"], errors="coerce").astype("Int64")
    if "business_type" not in out.columns:
        out["business_type"] = "Car"
    out = out.dropna(subset=["id_dealer_outlet","brand","city","name"])
    out = out[_valid_mask(out["latitude"], out["longitude"])]
    out = out[out["business_type"].astype(str).str.strip().str.lower()=="car"]
    out = out[["id_dealer_outlet","brand","business_type","city","name","latitude","longitude"]].drop_duplicates()
    return out.reset_index(drop=True)

def clean_visits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    cols = {c.lower().strip(): c for c in out.columns}
    emp = cols.get("nama karyawan") or cols.get("employee name") or cols.get("employee") or "Nama Karyawan"
    cli = cols.get("nama klien") or cols.get("client name") or "Nama Klien"
    dt = cols.get("tanggal datang") or cols.get("date time start") or "Tanggal Datang"
    latlon = cols.get("latitude & longitude datang") or cols.get("latitude start") or "Latitude & Longitude Datang"
    nik = cols.get("nomor induk karyawan") or cols.get("nik") or "Nomor Induk Karyawan"
    div = cols.get("divisi") or "Divisi"
    out = out.rename(columns={emp:"employee_name",cli:"client_name",dt:"visit_datetime",latlon:"latlon",nik:"nik",div:"divisi"})
    out["visit_datetime"] = pd.to_datetime(out["visit_datetime"], errors="coerce")
    lat, lon = zip(*out["latlon"].apply(split_latlon))
    out["lat"] = pd.Series(lat, index=out.index).apply(_to_float)
    out["long"] = pd.Series(lon, index=out.index).apply(_to_float)
    out.loc[~_valid_mask(out["lat"], out["long"]), ["lat","long"]] = np.nan
    out["nik"] = out.get("nik", pd.Series("", index=out.index)).astype(str)
    out["divisi"] = out.get("divisi", pd.Series("", index=out.index)).astype(str)
    out = out[~out["nik"].str.contains("deleted-", case=False, na=False)]
    out = out[~out["divisi"].str.contains("trainer", case=False, na=False)]
    out = out.dropna(subset=["employee_name"])
    out = out[["employee_name","client_name","visit_datetime","lat","long","nik","divisi"]]
    return out.reset_index(drop=True)

def assign_visits_to_dealers(visits: pd.DataFrame, dealers: pd.DataFrame, max_km=1.0) -> pd.DataFrame:
    if visits.empty or dealers.empty:
        v = visits.copy()
        v["matched_dealer_id"] = pd.NA
        v["matched_client"] = pd.NA
        return v
    d = dealers[["id_dealer_outlet","name","latitude","longitude","city","brand"]].dropna().copy()
    d = d[_valid_mask(d["latitude"], d["longitude"])]
    v = visits.copy()
    left = v.merge(d[["id_dealer_outlet","name"]], left_on="client_name", right_on="name", how="left")
    v["matched_dealer_id"] = left["id_dealer_outlet"]
    v["matched_client"] = left["name"] if "name" in left.columns else pd.NA
    mask = v["matched_dealer_id"].isna()
    if mask.any():
        base = v[mask].copy()
        base = base.dropna(subset=["lat","long"])
        base = base[_valid_mask(base["lat"], base["long"])]
        if not base.empty and not d.empty:
            dv = d[["id_dealer_outlet","name","latitude","longitude"]].dropna().copy()
            dv = dv[_valid_mask(dv["latitude"], dv["longitude"])]
            if not dv.empty:
                def nearest(row):
                    lat, lon = row["lat"], row["long"]
                    if not _valid_latlon(lat, lon):
                        return pd.Series([pd.NA, pd.NA])
                    dist = dv.apply(lambda r: _km(lat, lon, r.latitude, r.longitude), axis=1)
                    dv_ = dv.assign(_km=dist).sort_values("_km")
                    if dv_.empty or not np.isfinite(dv_["_km"].iloc[0]) or float(dv_["_km"].iloc[0]) > max_km:
                        return pd.Series([pd.NA, pd.NA])
                    return pd.Series([dv_.iloc[0]["id_dealer_outlet"], dv_.iloc[0]["name"]])
                nn = base.apply(nearest, axis=1)
                v.loc[base.index, "matched_dealer_id"] = nn.iloc[:,0].values
                v.loc[base.index, "matched_client"] = nn.iloc[:,1].values
    return v

def prepare_run_order(running_order: pd.DataFrame) -> pd.DataFrame:
    if running_order.empty:
        return pd.DataFrame(columns=["id_dealer_outlet","joined_dse","active_dse","nearest_end_date"])
    rm = running_order.rename(columns={"Dealer Id":"id_dealer_outlet","Dealer Name":"dealer_name","LMS Id":"joined_dse","IsActive":"active_dse","End Date":"End Date"})
    rm["id_dealer_outlet"] = pd.to_numeric(rm["id_dealer_outlet"], errors="coerce").astype("Int64")
    rm["joined_dse"] = pd.to_numeric(rm["joined_dse"], errors="coerce").astype("Int64")
    rm["active_dse"] = pd.to_numeric(rm["active_dse"], errors="coerce").astype("Int64")
    act = rm[rm["active_dse"]==1].copy()
    act["End Date"] = pd.to_datetime(act["End Date"], errors="coerce")
    ao = act.groupby(["id_dealer_outlet"], as_index=False)["End Date"].min().rename(columns={"End Date":"nearest_end_date"})
    grp = rm.groupby(["id_dealer_outlet"], as_index=False).agg(joined_dse=("joined_dse","count"), active_dse=("active_dse","sum"))
    out = grp.merge(ao, on="id_dealer_outlet", how="left")
    return out

def compute_availability(dealers: pd.DataFrame, ro_group: pd.DataFrame, location_detail: pd.DataFrame, need_cluster: pd.DataFrame) -> pd.DataFrame:
    df = dealers.copy()
    df = df.merge(ro_group, on="id_dealer_outlet", how="left")
    ld = location_detail.rename(columns={"City":"city","Cluster":"cluster"})
    df = df.merge(ld[["city","cluster"]], on="city", how="left")
    nc = need_cluster.rename(columns={"Cluster":"cluster","Brand":"brand","Daily_Gen":"daily_gen","Daily_Need":"daily_need","Delta":"delta","Tag":"availability"})
    nc["brand"] = nc["brand"].replace({"CHERY":"Chery","Kia":"KIA"})
    df = df.merge(nc[["cluster","brand","daily_gen","daily_need","delta","availability"]], on=["cluster","brand"], how="left")
    df["tag"] = np.where(df["nearest_end_date"].isna(),"Not Active","Active")
    return df

def cluster_by_bde(visits: pd.DataFrame, dealers: pd.DataFrame, only_name: str | None):
    if only_name in (None, "All"):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    vsel = visits[visits["employee_name"]==only_name][["visit_datetime","client_name","lat","long"]].dropna().copy()
    if vsel.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    vsel = vsel.rename(columns={"lat":"latitude","long":"longitude"})
    vsel = vsel[_valid_mask(vsel["latitude"], vsel["longitude"])]
    vsel["sales_name"] = only_name
    dbox = dealers.copy()
    dbox = dbox[_valid_mask(dbox["latitude"], dbox["longitude"])]
    dbox["sales_name"] = only_name
    if len(vsel) >= 4:
        X = vsel[["latitude","longitude"]].values.tolist()
        wcss = []
        for k in range(4, min(9, len(vsel))):
            km = KMeans(n_clusters=k, n_init="auto").fit(X)
            wcss.append(km.inertia_)
        if len(wcss) >= 2:
            kl = KneeLocator(range(4, 4+len(wcss)), wcss, curve="convex", direction="decreasing")
            n_cluster = kl.elbow if kl.elbow is not None else 4
        else:
            n_cluster = 4
        kmeans = KMeans(n_clusters=n_cluster, n_init="auto").fit(X)
        vsel["cluster"] = kmeans.labels_
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=["latitude","longitude"])
        centers["sales_name"] = only_name
        centers["cluster"] = range(len(centers))
        avails = []
        for i in range(len(centers)):
            latc, lonc = centers.loc[i,"latitude"], centers.loc[i,"longitude"]
            av_i = dbox.copy()
            av_i[f"dist_center_{i}"] = av_i.apply(lambda x: _km(latc, lonc, x.latitude, x.longitude), axis=1)
            avails.append(av_i)
        avail_df = pd.concat(avails) if avails else pd.DataFrame()
        return vsel, avail_df, centers
    vsel["cluster"] = 0
    latm = float(vsel["latitude"].mean()) if len(vsel) else np.nan
    lonm = float(vsel["longitude"].mean()) if len(vsel) else np.nan
    centers = pd.DataFrame([[latm, lonm, only_name, 0]], columns=["latitude","longitude","sales_name","cluster"])
    av_i = dbox.copy()
    av_i["dist_center_0"] = av_i.apply(lambda x: _km(latm, lonm, x.latitude, x.longitude), axis=1)
    return vsel, av_i, centers

def get_summary_data(visits: pd.DataFrame):
    if visits.empty:
        return pd.DataFrame(), pd.DataFrame()
    v = visits.copy()
    v["date"] = v["visit_datetime"].dt.date
    v = v.dropna(subset=["date"])
    v["month_year"] = pd.to_datetime(v["date"]).astype("datetime64[M]").astype(str)
    agg = v.groupby(["date","employee_name"], as_index=False)["client_name"].count().rename(columns={"client_name":"ctd_visit"})
    agg["avg_distance_km"] = 0.0
    agg["avg_time_between_minute"] = 0.0
    agg["avg_speed_kmpm"] = 0.0
    agg["month_year"] = pd.to_datetime(agg["date"]).astype("datetime64[M]").astype(str)
    return v, agg

def compute_all(bde_filter: str | None = None):
    sheets = get_sheets()
    dealers_raw = sheets.get("dealers", pd.DataFrame())
    visits_raw = sheets.get("visits", pd.DataFrame())
    location_detail = sheets.get("location_detail", pd.DataFrame())
    need_cluster = sheets.get("need_cluster", pd.DataFrame())
    running_order = sheets.get("running_order", pd.DataFrame())
    dealers = clean_dealers(dealers_raw)
    visits = clean_visits(visits_raw)
    visits = assign_visits_to_dealers(visits, dealers, max_km=1.0)
    ro_group = prepare_run_order(running_order)
    avail_df_merge = compute_availability(dealers, ro_group, location_detail, need_cluster)
    sum_df, avail_df, clust_df = cluster_by_bde(visits, dealers, bde_filter)
    if not avail_df.empty:
        dist_cols = [c for c in avail_df.columns if str(c).startswith("dist_center_")]
        if dist_cols:
            vals = avail_df[dist_cols].apply(pd.to_numeric, errors="coerce")
            mvals = vals.min(axis=1)
            for c in dist_cols:
                vc = pd.to_numeric(avail_df[c], errors="coerce")
                mask = np.isfinite(vc) & np.isfinite(mvals) & np.isclose(vc, mvals, rtol=0, atol=1e-9)
                avail_df[c] = np.where(mask, vc, np.nan)
            try:
                avail_df_merge = avail_df_merge.merge(avail_df[["id_dealer_outlet"]+dist_cols], on="id_dealer_outlet", how="left")
            except Exception:
                pass
    return {"dealers": dealers, "visits": visits, "sum_df": sum_df, "clust_df": clust_df, "avail_df_merge": avail_df_merge, "location_detail": sheets.get("location_detail", pd.DataFrame())}
