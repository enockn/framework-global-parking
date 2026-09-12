#!/usr/bin/env python3
"""Urban Onstreet Parking Ocupancy Prediction.

Run locally:
    python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

import copy, json, math, os
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk 
import streamlit as st

from framework_inference import DAY_ORDER, MONTH_NAMES, ForecastRequest, FrameworkPredictor

APP_TITLE="Urban Onstreet Parking Ocupancy Prediction"
CARTO_LIGHT="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
CARTO_DARK="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
st.set_page_config(page_title=APP_TITLE,page_icon="🅿️",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding-top:1rem;max-width:1700px}.hero{padding:1.15rem 1.4rem;border-radius:20px;background:linear-gradient(120deg,#07111f,#132c4b 55%,#0f766e);color:#fff;margin-bottom:.9rem;box-shadow:0 12px 36px rgba(15,23,42,.18)}.hero h1{margin:0;font-size:2rem}.hero p{margin:.25rem 0 0;opacity:.88}.badge{display:inline-block;padding:.22rem .55rem;border-radius:999px;font-size:.82rem;font-weight:700;background:#e2e8f0;color:#0f172a}.small{font-size:.86rem;color:#64748b}.legend{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:.4rem 0 .8rem}.dot{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:5px}
</style>
""",unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_predictor(path): return FrameworkPredictor(path)
@st.cache_data(show_spinner=False)
def load_geojson(path): return json.loads((Path(path)/"parking_spaces.geojson").read_text(encoding="utf-8"))

def bundle_default():
    """Resolve the serving bundle without exposing a filesystem control publicly."""
    try:
        secret_value = st.secrets.get("FRAMEWORK_BUNDLE_DIR", None)
    except Exception:
        secret_value = None
    return str(secret_value or os.environ.get("FRAMEWORK_BUNDLE_DIR", "framework_serving"))

def hhmm(x:float):
    h=int(x); m=int(round((x-h)*60)); return f"{h:02d}:{m:02d}"

def map_table(gj):
    return pd.DataFrame([f.get("properties",{}) for f in gj.get("features",[])])

def filter_spaces(df,road,section,ext,geoms,types,ways):
    x=df.copy()
    if road!="All streets": x=x[x.Street_Name.astype(str).eq(road)]
    if section!="All road sections": x=x[x.road_section_uid.astype(str).eq(section)]
    if ext!="All ext_id": x=x[pd.to_numeric(x.ext_id,errors="coerce").eq(int(ext))]
    if geoms: x=x[x.type_geom.astype(str).isin(geoms)]
    if types: x=x[x.parking_type.astype(str).isin(types)]
    if ways: x=x[x.wayside.astype(str).isin(ways)]
    return x

def bounds_for(gj,uids):
    xs=[]; ys=[]
    for f in gj.get("features",[]):
        p=f.get("properties",{})
        if int(p.get("space_uid",-1)) not in uids: continue
        for key in ["min_lon","max_lon"]:
            try: xs.append(float(p[key]))
            except: pass
        for key in ["min_lat","max_lat"]:
            try: ys.append(float(p[key]))
            except: pass
    return (min(xs),min(ys),max(xs),max(ys)) if xs and ys else None

def view_for(bounds,world=False):
    if world or bounds is None: return pdk.ViewState(latitude=18,longitude=5,zoom=1.15,pitch=0,bearing=0)
    minx,miny,maxx,maxy=bounds; lon=(minx+maxx)/2; lat=(miny+maxy)/2
    span=max(maxx-minx,maxy-miny,1e-5); zoom=float(np.clip(8.9-math.log2(span*111+1e-6),3,18))
    return pdk.ViewState(latitude=lat,longitude=lon,zoom=zoom,pitch=0,bearing=0)

def render_geo(gj,selected,pred,show_unselected=True):
    pm={int(r.ext_id):r._asdict() for r in pred.itertuples(index=False)} if len(pred) else {}
    out={"type":"FeatureCollection","features":[]}
    for f0 in gj.get("features",[]):
        f=copy.deepcopy(f0); p=f.setdefault("properties",{}); uid=int(p.get("space_uid",-1)); ext=int(p.get("ext_id",-1)); sel=uid in selected
        if not sel and not show_unselected: continue
        r=pm.get(ext)
        if r is not None:
            p["fill_color"]=list(r["fill_color"]); p["line_color"]=[255,255,255,210]
            p["predicted_pct"]=f"{100*r['predicted_occupancy']:.1f}%"
            p["PI90"]=f"{100*r['PI90_low']:.1f}% – {100*r['PI90_high']:.1f}%"; p["PI95"]=f"{100*r['PI95_low']:.1f}% – {100*r['PI95_high']:.1f}%"
            if np.isfinite(r.get("actual_occupancy",np.nan)):
                p["actual_pct"]=f"{100*r['actual_occupancy']:.1f}%"; p["abs_error_pct"]=f"{100*r['absolute_error']:.2f} pp"
            else:
                p["actual_pct"]="Future — not observed"; p["abs_error_pct"]="—"
            p["source"]=str(r.get("framework_source","Framework")); p["mode"]=str(r.get("forecast_mode","Forecast"))
            p["expected_occupied"]=f"{r['expected_occupied_spaces']:.1f}"; p["expected_available"]=f"{r['expected_available_spaces']:.1f}"
        elif sel:
            p["fill_color"]=[14,165,233,170]; p["line_color"]=[255,255,255,200]; p["predicted_pct"]="Not yet predicted"; p["actual_pct"]="—"; p["PI90"]="—"; p["PI95"]="—"; p["abs_error_pct"]="—"; p["source"]="—"; p["mode"]="Selected"; p["expected_occupied"]="—"; p["expected_available"]="—"
        else:
            p["fill_color"]=[100,116,139,34]; p["line_color"]=[148,163,184,70]; p["predicted_pct"]="Background"; p["actual_pct"]="—"; p["PI90"]="—"; p["PI95"]="—"; p["abs_error_pct"]="—"; p["source"]="—"; p["mode"]="Background"; p["expected_occupied"]="—"; p["expected_available"]="—"
        out["features"].append(f)
    return out

def deck(gj,view,dark):
    layer=pdk.Layer("GeoJsonLayer",data=gj,pickable=True,auto_highlight=True,stroked=True,filled=True,get_fill_color="properties.fill_color",get_line_color="properties.line_color",get_line_width=1.25,line_width_min_pixels=.8,opacity=.9,highlight_color=[56,189,248,160])
    tip={"html":"""<div style='font-family:Arial;min-width:300px'><b style='font-size:15px'>{Street_Name}</b><br/><b>ext_id:</b> {ext_id}<br/><b>Road section:</b> {road_section_uid}<br/><b>Type:</b> {parking_type}<br/><b>Geometry:</b> {type_geom}<br/><b>Wayside:</b> {wayside}<hr style='opacity:.3'/><b>Predicted occupancy:</b> {predicted_pct}<br/><b>Actual occupancy:</b> {actual_pct}<br/><b>Absolute error:</b> {abs_error_pct}<br/><b>90% interval:</b> {PI90}<br/><b>95% interval:</b> {PI95}<br/><b>Expected occupied:</b> {expected_occupied}<br/><b>Expected free:</b> {expected_available}<br/><b>Framework source:</b> {source}<br/><b>Mode:</b> {mode}</div>""","style":{"backgroundColor":"#0f172a","color":"white","borderRadius":"10px"}}
    return pdk.Deck(layers=[layer],initial_view_state=view,map_style=CARTO_DARK if dark else CARTO_LIGHT,tooltip=tip)

def next_matching_date(year:int,month:int,weekday_name:str):
    target=DAY_ORDER.index(weekday_name); d=date(year,month,1)
    shift=(target-d.weekday())%7
    return d+timedelta(days=shift)

st.markdown(f"<div class='hero'><h1>🅿️ {APP_TITLE}</h1><p>Framework occupancy forecasting • historical validation • future-only inference • global spatial interface</p></div>",unsafe_allow_html=True)

bundle_dir=bundle_default()
bundle=Path(bundle_dir)

with st.sidebar:
    st.caption("Framework serving assets loaded from the deployed repository.")
if not (bundle/"serving_metadata.json").exists():
    st.error("Serving bundle not found. Run the preparation command shown below, then refresh.")
    st.code('python prepare_framework_platform.py --events "merged_parking_data.csv" --spaces "mldf_clean_merge.csv" --framework-results "outputs_pd_evaluation_v9" --outdir "framework_serving"\n\npython -m streamlit run streamlit_app.py',language="powershell")
    st.stop()

with st.spinner("Loading Framework assets …"):
    predictor=load_predictor(str(bundle.resolve())); gj=load_geojson(str(bundle.resolve())); sdf=map_table(gj)

with st.sidebar:
    st.divider(); st.subheader("1 · Location")
    roads=["All streets"]+sorted(sdf.Street_Name.dropna().astype(str).unique()); road=st.selectbox("Street name",roads)
    s1=sdf if road=="All streets" else sdf[sdf.Street_Name.astype(str).eq(road)]
    sections=["All road sections"]+sorted(s1.road_section_uid.dropna().astype(str).unique()); section=st.selectbox("Road section UID",sections)
    s2=s1 if section=="All road sections" else s1[s1.road_section_uid.astype(str).eq(section)]
    exts=["All ext_id"]+[str(x) for x in sorted(pd.to_numeric(s2.ext_id,errors="coerce").dropna().astype(int).unique())]; ext=st.selectbox("ext_id",exts)
    geoms=st.multiselect("Parking geometry (type_geon)",sorted(sdf.type_geom.dropna().astype(str).unique()))
    types=st.multiselect("Parking type",sorted(sdf.parking_type.dropna().astype(str).unique()))
    ways=st.multiselect("Wayside",sorted(sdf.wayside.dropna().astype(str).unique()))

    st.subheader("2 · Forecast target")
    mode=st.radio("Time selection",["Exact date & time","Month + weekday + time"],horizontal=False)
    if mode=="Exact date & time":
        default_date=(predictor.data_end+pd.Timedelta(days=30)).date()
        qdate=st.date_input("Target date",value=default_date)
        qtime=st.time_input("Target time",value=dtime(12,0),step=900)
        target=pd.Timestamp(datetime.combine(qdate,qtime))
    else:
        year=st.number_input("Forecast year",min_value=2022,max_value=2100,value=max(pd.Timestamp.now().year,predictor.data_end.year+1),step=1)
        month=st.selectbox("Month",list(range(1,13)),format_func=lambda x:MONTH_NAMES[x])
        weekday=st.selectbox("Day of week",DAY_ORDER)
        qtime=st.time_input("Time",value=dtime(12,0),step=900)
        qdate=next_matching_date(int(year),int(month),weekday)
        target=pd.Timestamp(datetime.combine(qdate,qtime))
        st.caption(f"Representative target date: {target:%A, %d %B %Y}")
    horizons=predictor.available_horizons(); horizon=st.selectbox("Framework horizon",horizons,index=min(2,len(horizons)-1),format_func=lambda x:f"{x} min")
    origin=target-pd.Timedelta(minutes=int(horizon))
    st.caption(f"Derived target: {target:%B} · {target:%A} · {target:%H:%M}")
    st.caption(f"Framework origin: {origin:%Y-%m-%d %H:%M}")

    st.subheader("3 · Map")
    auto_zoom=st.toggle("Auto-zoom",True); show_unselected=st.toggle("Show background spaces",True); dark=st.toggle("Dark map",False)
    reset=st.button("🌍 Global view",use_container_width=True); go=st.button("Predict occupancy",type="primary",use_container_width=True)

filtered=filter_spaces(sdf,road,section,ext,geoms,types,ways)
selected_uids=set(pd.to_numeric(filtered.space_uid,errors="coerce").dropna().astype(int)); selected_ext=sorted(pd.to_numeric(filtered.ext_id,errors="coerce").dropna().astype(int).unique())
if "forecast" not in st.session_state: st.session_state.forecast=pd.DataFrame(); st.session_state.request=None
if go:
    if not selected_ext: st.warning("No parking spaces match the current filters.")
    elif target.day_name()=="Sunday": st.warning("Sunday was not part of the observed Framework operating support. Choose Monday–Saturday for research-consistent inference.")
    else:
        with st.spinner(f"Running Framework inference for {len(selected_ext):,} locations …"):
            req=ForecastRequest(target_datetime=target,horizon_min=int(horizon))
            try:
                st.session_state.forecast=predictor.predict(selected_ext,req)
                st.session_state.request=req
            except Exception as exc:
                st.session_state.forecast=pd.DataFrame()
                st.session_state.request=None
                st.error(f"Framework inference failed: {exc}")
                st.exception(exc)

pred=st.session_state.forecast.copy()
if len(pred): pred=pred[pred.ext_id.isin(selected_ext)].copy()

if st.session_state.request is not None:
    rq=st.session_state.request; historical=(pd.Timestamp(rq.target_datetime)>=predictor.data_start and pd.Timestamp(rq.target_datetime)<=predictor.data_end)
    if historical:
        st.success(f"Historical validation mode — actual occupancy is available for {pd.Timestamp(rq.target_datetime):%Y-%m-%d %H:%M} and is shown beside the Framework forecast.")
    else:
        st.info(f"Future forecast mode — {pd.Timestamp(rq.target_datetime):%Y-%m-%d %H:%M} lies outside the observed dataset. Only the Framework forecast is shown; no actual value is fabricated.")
else:
    st.caption("Choose location and target time, then press Predict occupancy.")

# summary
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Parking polygons",f"{len(filtered):,}"); c2.metric("ext_id",f"{len(selected_ext):,}")
if len(pred):
    C=pd.to_numeric(pred.capacity,errors="coerce").fillna(1).clip(lower=1).to_numpy(float); P=pred.predicted_occupancy.to_numpy(float); cap=C.sum(); occ=np.average(P,weights=C); free=np.sum((1-P)*C)
    c3.metric("Legal capacity",f"{int(round(cap)):,}"); c4.metric("Predicted occupancy",f"{100*occ:.1f}%"); c5.metric("Expected free spaces",f"{free:.0f}")
else:
    c3.metric("Legal capacity","—"); c4.metric("Predicted occupancy","—"); c5.metric("Expected free spaces","—")

st.markdown("<div class='legend'><span><span class='dot' style='background:#16a34a'></span><b>Green</b> &lt;5%</span><span><span class='dot' style='background:#f59e0b'></span><b>Orange</b> 5–50%</span><span><span class='dot' style='background:#dc2626'></span><b>Red</b> &gt;50%</span></div>",unsafe_allow_html=True)
render=render_geo(gj,selected_uids,pred,show_unselected)
world=reset or not auto_zoom or len(selected_uids)==0
st.pydeck_chart(deck(render,view_for(bounds_for(gj,selected_uids),world),dark),use_container_width=True,height=720)

if len(pred):
    historical=pred.actual_occupancy.notna().any()
    if historical:
        valid=pred.dropna(subset=["actual_occupancy"])
        mae=float(np.mean(np.abs(valid.actual_occupancy-valid.predicted_occupancy))) if len(valid) else np.nan
        rmse=float(np.sqrt(np.mean((valid.actual_occupancy-valid.predicted_occupancy)**2))) if len(valid) else np.nan
        a,b,c=st.columns(3); a.metric("Historical MAE",f"{100*mae:.3f} pp"); b.metric("Historical RMSE",f"{100*rmse:.3f} pp"); c.metric("Validated locations",f"{len(valid):,}")

    st.subheader("Location-level Framework results")
    cols=["Street_Name","road_section_uid","ext_id","parking_type","type_geom","wayside","locality_km","capacity","predicted_occupancy","PI90_low","PI90_high","PI95_low","PI95_high","expected_occupied_spaces","expected_available_spaces","framework_source","historical_state_source","forecast_mode"]
    if historical: cols.insert(9,"actual_occupancy"); cols.insert(10,"absolute_error")
    tab=pred[cols].sort_values("predicted_occupancy",ascending=False)
    st.dataframe(tab,use_container_width=True,hide_index=True,column_config={"predicted_occupancy":st.column_config.ProgressColumn("Predicted occupancy",min_value=0,max_value=1,format="%.3f"),"actual_occupancy":st.column_config.NumberColumn("Actual occupancy",format="%.3f"),"absolute_error":st.column_config.NumberColumn("Absolute error",format="%.4f")})
    st.download_button("Download forecast CSV",tab.to_csv(index=False).encode(),file_name="framework_parking_forecast.csv",mime="text/csv")

with st.expander("Scientific forecast semantics"):
    st.markdown(fr"""
**The platform does not train a replacement model.** It uses the exact saved **Framework** transition forecasts from the research pipeline.

For a genuinely future target, the current parking state is not yet observable. The platform therefore marginalizes Framework behaviour over historically comparable parking states:

\[
\widehat O_{{j}}(t+h)=\widehat E[O_j(t)\mid j,\,m,\,d,\,\tau]+\widehat E[\Delta^{{Framework}}_j(h)\mid \text{{analogous historical states}}].
\]

The first term is estimated only from historical states observed at **real `arrival_time` values**. For every event row, month, weekday and hour are derived directly from `merged_parking_data.csv -> arrival_time`. The second term uses the exact held-out Framework transition \(M4-O(t)\), not a newly fitted surrogate.

For timestamps inside the dataset, actual occupancy is reconstructed from observed arrivals and departures and shown for accuracy checking. For dates outside the dataset, actual occupancy does not exist yet and is therefore **never fabricated or displayed**.

Observed data range: **{predictor.data_start} to {predictor.data_end}**.
    """)
