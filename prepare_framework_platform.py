"""
Prepare the Framework Global Parking Intelligence Platform serving bundle.

Scientific principle
--------------------
This deployment builder does NOT fit a new surrogate forecasting model.
It preserves the Framework developed in the research pipeline by using:

1. historical occupancy states reconstructed directly from merged_parking_data.csv;
2. month / day_of_week / hour_of_day derived line-by-line from each event's
   arrival_time (e.g. 1/25/2023 17:24 -> January, Wednesday, 17:24);
3. the exact held-out Framework M4 predictions written by the research pipeline;
4. empirical analogue marginalisation for genuinely future dates, where the
   future current parking state is necessarily unobserved.

For a future query, the platform estimates the expected current occupancy from
historically comparable arrival-time states, then applies the transition delta
observed from exact Framework M4 analogue states.  No future actual occupancy is
used or displayed.

For historical timestamps inside the source data, the platform can reconstruct
actual occupancy from the event index so the forecast can be checked against
reality.  If the timestamp is one of the held-out Framework test origins, the
exact saved Framework M4 prediction is returned.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely import wkt

try:
    from shapely.validation import make_valid
except Exception:
    make_valid = None

ALLOWED_TYPES = ["Payantes", "Livraison", "PMR", "Gratuites"]
ALLOWED_GEOMETRIES = ["LONGITUDINAL", "EPI+75", "BATAILLE"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def clean_ext_id(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    token = s.astype(str).str.strip().str.extract(r"(-?\d+)", expand=False)
    return pd.to_numeric(token, errors="coerce").astype("Int64")


def robust_datetime(s: pd.Series) -> pd.Series:
    """Parse timestamps such as 1/25/2023 17:24 robustly and explicitly."""
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
    except TypeError:  # pandas < 2.0
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=False)


def parse_locality(v) -> float:
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    m = re.search(r"[-+]?\d*\.?\d+", str(v).replace(",", "."))
    return float(m.group()) if m else np.nan


def mode_or_unknown(s: pd.Series, default="UNKNOWN") -> str:
    x = s.dropna().astype(str).str.strip()
    x = x[x.ne("")]
    if x.empty:
        return default
    m = x.mode()
    return str(m.iloc[0] if len(m) else x.iloc[0])


def canon(v, allowed):
    if pd.isna(v):
        return "UNKNOWN"
    raw = str(v).strip()
    lut = {x.upper(): x for x in allowed}
    return lut.get(raw.upper(), raw)


def safe_geometry(text):
    try:
        g = wkt.loads(str(text))
        if not g.is_valid:
            g = make_valid(g) if make_valid is not None else g.buffer(0)
        if g.is_empty:
            return None
        if g.geom_type == "GeometryCollection":
            from shapely.ops import unary_union
            polys = [q for q in g.geoms if q.geom_type in {"Polygon", "MultiPolygon"}]
            if not polys:
                return None
            g = unary_union(polys)
        return g if g.geom_type in {"Polygon", "MultiPolygon"} else None
    except Exception:
        return None


def load_spaces(spaces_path: Path, results_dir: Path):
    hdr = pd.read_csv(spaces_path, nrows=0).columns.tolist()
    ptype = "type" if "type" in hdr else "parking_type"
    pgeom = "type_geom" if "type_geom" in hdr else "type_geom"
    locality = "locality" if "locality" in hdr else "locality_km"
    needed = ["ext_id", "spot_count", "wayside", "Street_Name", "road_section_uid", "geom", ptype, pgeom, locality]
    missing = [c for c in needed if c not in hdr]
    if missing:
        raise ValueError(f"Parking-space file is missing required columns: {missing}")

    sp = pd.read_csv(spaces_path, usecols=needed, low_memory=False)
    sp["ext_id"] = clean_ext_id(sp["ext_id"])
    sp = sp.dropna(subset=["ext_id", "geom"]).copy()
    sp["ext_id"] = sp["ext_id"].astype(int)
    sp["spot_count"] = pd.to_numeric(sp["spot_count"], errors="coerce")
    sp = sp[np.isfinite(sp["spot_count"]) & (sp["spot_count"] > 0)].copy()
    sp["parking_type"] = sp[ptype].map(lambda x: canon(x, ALLOWED_TYPES))
    sp["type_geom"] = sp[pgeom].map(lambda x: canon(x, ALLOWED_GEOMETRIES))
    sp["wayside"] = sp["wayside"].astype("string").str.strip().str.upper().fillna("UNKNOWN")
    sp["Street_Name"] = sp["Street_Name"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip().fillna("UNKNOWN ROAD")
    sp["road_section_uid"] = sp["road_section_uid"].astype("string").str.strip().fillna("UNKNOWN SECTION")
    sp["locality_km"] = sp[locality].map(parse_locality)
    sp = sp[sp["parking_type"].isin(ALLOWED_TYPES) & sp["type_geom"].isin(ALLOWED_GEOMETRIES)].copy()
    if sp.empty:
        raise RuntimeError("No parking polygons remain after Framework type/geometry filtering.")

    cap = sp.groupby("ext_id", sort=False)["spot_count"].max().round().astype(int).rename("capacity")
    raw_ctx = sp.groupby("ext_id", as_index=False).agg(
        Street_Name=("Street_Name", mode_or_unknown),
        road_section_uid=("road_section_uid", mode_or_unknown),
        parking_type=("parking_type", mode_or_unknown),
        type_geom=("type_geom", mode_or_unknown),
        wayside=("wayside", mode_or_unknown),
        locality_km=("locality_km", "median"),
    ).merge(cap.reset_index(), on="ext_id", how="inner")

    # Prefer the exact research context where available, but preserve raw geometry fields.
    cp = results_dir / "parking_context_by_ext_id.csv"
    if cp.exists():
        research = pd.read_csv(cp, low_memory=False)
        research["ext_id"] = clean_ext_id(research["ext_id"])
        research = research.dropna(subset=["ext_id"]).copy(); research["ext_id"] = research["ext_id"].astype(int)
        ren = {}
        if "type" in research and "parking_type" not in research: ren["type"] = "parking_type"
        if "type_geom" in research and "type_geom" not in research: ren["type_geom"] = "type_geom"
        if "locality" in research and "locality_km" not in research: ren["locality"] = "locality_km"
        research = research.rename(columns=ren)
        cols = [c for c in ["ext_id", "Street_Name", "road_section_uid", "parking_type", "type_geom", "wayside", "locality_km"] if c in research]
        ctx = raw_ctx.merge(research[cols], on="ext_id", how="left", suffixes=("_raw", ""))
        for c in ["Street_Name", "road_section_uid", "parking_type", "type_geom", "wayside", "locality_km"]:
            rc = c + "_raw"
            if c not in ctx and rc in ctx:
                ctx[c] = ctx[rc]
            elif rc in ctx:
                ctx[c] = ctx[c].where(ctx[c].notna() & ctx[c].astype(str).ne(""), ctx[rc])
            if rc in ctx:
                ctx = ctx.drop(columns=rc)
    else:
        ctx = raw_ctx.copy()

    ctx["locality_km"] = pd.to_numeric(ctx["locality_km"], errors="coerce")
    med = float(ctx["locality_km"].median()) if ctx["locality_km"].notna().any() else 0.0
    ctx["locality_km"] = ctx["locality_km"].fillna(med)
    for c in ["Street_Name", "road_section_uid", "parking_type", "type_geom", "wayside"]:
        ctx[c] = ctx[c].astype("string").fillna("UNKNOWN").astype(str)

    valid_ids = set(ctx["ext_id"].astype(int))
    sp = sp[sp["ext_id"].isin(valid_ids)].copy()

    # GeoJSON preserves every distinct polygon.
    features = []
    seen = set(); bad = 0
    for r in sp.itertuples(index=False):
        key = (int(r.ext_id), str(r.geom), str(r.parking_type), str(r.type_geom), str(r.wayside))
        if key in seen:
            continue
        seen.add(key)
        g = safe_geometry(r.geom)
        if g is None:
            bad += 1; continue
        from shapely.geometry import mapping
        minx, miny, maxx, maxy = g.bounds
        cen = g.centroid
        features.append({
            "type": "Feature",
            "geometry": mapping(g),
            "properties": {
                "space_uid": len(features), "ext_id": int(r.ext_id),
                "Street_Name": str(r.Street_Name), "road_section_uid": str(r.road_section_uid),
                "parking_type": str(r.parking_type), "type_geom": str(r.type_geom),
                "wayside": str(r.wayside), "locality_km": None if not np.isfinite(r.locality_km) else float(r.locality_km),
                "centroid_lon": float(cen.x), "centroid_lat": float(cen.y),
                "min_lon": float(minx), "min_lat": float(miny), "max_lon": float(maxx), "max_lat": float(maxy),
            }
        })
    return ctx.drop_duplicates("ext_id").reset_index(drop=True), {"type":"FeatureCollection","features":features}, {"n_features":len(features),"n_bad_geometry":bad}


def load_events(events_path: Path, valid_ids: set[int]):
    hdr = pd.read_csv(events_path, nrows=0).columns.tolist() if events_path.suffix.lower() not in {".parquet", ".pq"} else None
    cols = ["ext_id", "arrival_time", "departure_time"]
    if events_path.suffix.lower() in {".parquet", ".pq"}:
        ev = pd.read_parquet(events_path, columns=cols)
    else:
        missing = [c for c in cols if c not in hdr]
        if missing: raise ValueError(f"Event file missing required columns: {missing}")
        ev = pd.read_csv(events_path, usecols=cols, low_memory=False)
    ev["ext_id"] = clean_ext_id(ev["ext_id"])
    ev["arrival_time"] = robust_datetime(ev["arrival_time"])
    ev["departure_time"] = robust_datetime(ev["departure_time"])
    ev = ev.dropna(subset=["ext_id", "arrival_time"]).copy(); ev["ext_id"] = ev["ext_id"].astype(int)
    ev = ev[ev["ext_id"].isin(valid_ids)].copy()
    ev = ev[ev["departure_time"].notna() & (ev["departure_time"] > ev["arrival_time"])].copy()

    # REQUIRED: derive the calendar variables line-by-line from arrival_time.
    at = pd.DatetimeIndex(ev["arrival_time"])
    ev["month"] = at.month.astype(np.int8)
    ev["day_of_week"] = at.day_name().astype(str)
    ev["hour_of_day"] = (at.hour + at.minute / 60.0).astype(np.float32)
    ev["hour_bin"] = (np.floor(ev["hour_of_day"].to_numpy(float) * 4.0) / 4.0).astype(np.float32)

    meta = {
        "data_start": str(ev["arrival_time"].min()), "data_end": str(ev["arrival_time"].max()),
        "arrival_months": sorted(int(x) for x in ev["month"].dropna().unique()),
        "arrival_days_of_week": [x for x in DAY_ORDER if x in set(ev["day_of_week"].dropna().astype(str))],
        "arrival_hours": sorted(float(x) for x in ev["hour_bin"].dropna().unique() if 7 <= float(x) < 19),
        "n_exact_history_events": int(len(ev)),
        "calendar_source": "merged_parking_data.arrival_time line-by-line",
    }
    return ev, meta


def build_event_index(ev: pd.DataFrame, ctx: pd.DataFrame, out: Path):
    """Compact exact occupancy index for arbitrary historical timestamps."""
    ids = sorted(set(ctx["ext_id"].astype(int)) & set(ev["ext_id"].astype(int)))
    arr_parts=[]; dep_parts=[]; offsets=[0]
    grp=ev.groupby("ext_id",sort=False)
    for ext in ids:
        g=grp.get_group(ext)
        arr=np.sort(g["arrival_time"].astype("int64").to_numpy())
        dep=np.sort(g["departure_time"].astype("int64").to_numpy())
        arr_parts.append(arr); dep_parts.append(dep); offsets.append(offsets[-1]+len(arr))
    arrivals=np.concatenate(arr_parts) if arr_parts else np.array([],dtype=np.int64)
    departures=np.concatenate(dep_parts) if dep_parts else np.array([],dtype=np.int64)
    np.savez_compressed(out, ext_ids=np.asarray(ids,np.int64), offsets=np.asarray(offsets,np.int64), arrivals_ns=arrivals, departures_ns=departures)


def historical_reference_from_arrivals(ev: pd.DataFrame, ctx: pd.DataFrame, samples_per_ext: int, seed: int):
    """Historical occupancy states sampled ONLY at real event arrival timestamps."""
    cap=ctx.set_index("ext_id")["capacity"].astype(int).to_dict()
    rows=[]; grp=ev.groupby("ext_id",sort=False)
    ids=[int(x) for x in ctx["ext_id"] if int(x) in grp.groups]
    for i,ext in enumerate(ids,1):
        g=grp.get_group(ext).sort_values("arrival_time")
        # each candidate timestamp is literally an arrival_time from a data row
        cand=g[["arrival_time","month","day_of_week","hour_of_day","hour_bin"]].drop_duplicates("arrival_time")
        if samples_per_ext>0 and len(cand)>samples_per_ext:
            # Keep broad temporal coverage while retaining only REAL arrival_time rows.
            idx=np.linspace(0,len(cand)-1,samples_per_ext).astype(int)
            cand=cand.iloc[np.unique(idx)].copy()
        arr=np.sort(g["arrival_time"].astype("int64").to_numpy()); dep=np.sort(g["departure_time"].astype("int64").to_numpy())
        tns=cand["arrival_time"].astype("int64").to_numpy()
        count=np.searchsorted(arr,tns,side="right")-np.searchsorted(dep,tns,side="right")
        C=max(int(cap[ext]),1); occ=np.clip(count,0,C)/C
        q=cand.copy(); q["ext_id"]=ext; q["occupancy"]=occ; q["capacity"]=C
        rows.append(q)
        if i%500==0: print(f"[history] processed {i:,}/{len(ids):,} ext_id")
    if not rows: raise RuntimeError("No historical arrival-time states could be constructed.")
    return pd.concat(rows,ignore_index=True)


def build_climatology(hist: pd.DataFrame, ctx: pd.DataFrame):
    x=hist.merge(ctx[["ext_id","road_section_uid","Street_Name","parking_type","type_geom","wayside","locality_km"]],on="ext_id",how="left")
    # Fine ext_id profile from actual arrival-time states.
    keys=["ext_id","month","day_of_week","hour_bin"]
    def grouped(keys_):
        return (x.groupby(keys_,sort=False)["occupancy"]
                .agg(n="size",mean_occupancy="mean",median_occupancy="median",
                     q05=lambda s: s.quantile(.05),q25=lambda s: s.quantile(.25),
                     q75=lambda s: s.quantile(.75),q95=lambda s: s.quantile(.95))
                .reset_index())
    fine=grouped(keys)
    # ext overall fallback
    ext=grouped(["ext_id"])
    # road-section temporal fallback
    section=grouped(["road_section_uid","month","day_of_week","hour_bin"])
    # context fallback
    context=grouped(["parking_type","type_geom","wayside","month","day_of_week","hour_bin"])
    global_cal=grouped(["month","day_of_week","hour_bin"])
    return fine,ext,section,context,global_cal


def load_framework_library(results_dir: Path, ctx: pd.DataFrame):
    p=None
    for n in ["test_predictions.parquet","test_predictions.csv.gz","test_predictions.csv"]:
        if (results_dir/n).exists(): p=results_dir/n; break
    if p is None: raise FileNotFoundError(f"No Framework test_predictions file found in {results_dir}")
    d=pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p,low_memory=False)
    need={"ext_id","time","horizon_min","occ_now","M4"}
    missing=need-set(d.columns)
    if missing: raise ValueError(f"Framework prediction file lacks: {sorted(missing)}")
    d["ext_id"]=clean_ext_id(d["ext_id"]); d=d.dropna(subset=["ext_id"]).copy(); d["ext_id"]=d["ext_id"].astype(int)
    d["time"]=robust_datetime(d["time"]); d=d.dropna(subset=["time","M4","occ_now","horizon_min"])
    d=d[d["ext_id"].isin(set(ctx["ext_id"]))].copy()
    t=pd.DatetimeIndex(d["time"])
    d["month"]=t.month.astype(np.int8); d["day_of_week"]=t.day_name().astype(str)
    d["hour_of_day"]=(t.hour+t.minute/60.0).astype(np.float32); d["hour_bin"]=(np.floor(d["hour_of_day"]*4)/4).astype(np.float32)
    d["delta_framework"]=pd.to_numeric(d["M4"],errors="coerce")-pd.to_numeric(d["occ_now"],errors="coerce")
    keep=["ext_id","time","horizon_min","month","day_of_week","hour_of_day","hour_bin","occ_now","M4","delta_framework"]
    for c in ["y","pchange_PD","persist_rate","M4_w_PERSIST","M4_w_M2","M4_w_HURDLE_PD","M4_w_M3"]:
        if c in d: keep.append(c)
    d=d[keep].merge(ctx[["ext_id","road_section_uid","Street_Name","parking_type","type_geom","wayside","locality_km"]],on="ext_id",how="left")
    return d


def framework_error_quantiles(lib: pd.DataFrame):
    out={}
    if "y" in lib:
        for h,g in lib.groupby("horizon_min"):
            e=np.abs(pd.to_numeric(g["y"],errors="coerce")-pd.to_numeric(g["M4"],errors="coerce")); e=e[np.isfinite(e)]
            if len(e): out[str(int(h))]={"q90":float(np.quantile(e,.90)),"q95":float(np.quantile(e,.95))}
    return out


def save_df(df: pd.DataFrame, path_no_suffix: Path):
    try:
        df.to_parquet(path_no_suffix.with_suffix(".parquet"),index=False)
        return str(path_no_suffix.with_suffix(".parquet").name)
    except Exception:
        df.to_csv(path_no_suffix.with_suffix(".csv.gz"),index=False,compression="gzip")
        return str(path_no_suffix.with_suffix(".csv.gz").name)


def main():
    ap=argparse.ArgumentParser(description="Build exact-Framework analogue serving bundle")
    ap.add_argument("--events",default="merged_parking_data.csv")
    ap.add_argument("--spaces",default="mldf_clean_merge.csv")
    ap.add_argument("--framework-results",default="outputs_pd_evaluation_v9")
    ap.add_argument("--outdir",default="framework_serving")
    ap.add_argument("--samples-per-ext",type=int,default=360,help="Real arrival timestamps retained per ext_id for historical climatology")
    args=ap.parse_args()
    events=Path(args.events); spaces=Path(args.spaces); results=Path(args.framework_results); outdir=Path(args.outdir)
    for p in [events,spaces,results]:
        if not p.exists(): raise FileNotFoundError(p)
    outdir.mkdir(parents=True,exist_ok=True)

    print("[1/7] Loading parking-space geometry and Framework context ...")
    ctx,geojson,gdiag=load_spaces(spaces,results)
    (outdir/"parking_spaces.geojson").write_text(json.dumps(geojson,ensure_ascii=False),encoding="utf-8")
    save_df(ctx,outdir/"parking_context")

    print("[2/7] Loading events and deriving month/day/hour from arrival_time line-by-line ...")
    ev,emeta=load_events(events,set(ctx["ext_id"].astype(int)))

    print("[3/7] Building exact historical event index for past-time accuracy checks ...")
    build_event_index(ev,ctx,outdir/"event_index.npz")

    print("[4/7] Building historical occupancy states at REAL arrival_time values ...")
    hist=historical_reference_from_arrivals(ev,ctx,args.samples_per_ext,42)
    hist_file=save_df(hist,outdir/"historical_arrival_state_reference")

    print("[5/7] Building multi-resolution historical climatology ...")
    fine,ext,section,context,global_cal=build_climatology(hist,ctx)
    files={
        "climatology_ext_calendar":save_df(fine,outdir/"climatology_ext_calendar"),
        "climatology_ext":save_df(ext,outdir/"climatology_ext"),
        "climatology_section_calendar":save_df(section,outdir/"climatology_section_calendar"),
        "climatology_context_calendar":save_df(context,outdir/"climatology_context_calendar"),
        "climatology_global_calendar":save_df(global_cal,outdir/"climatology_global_calendar"),
        "historical_reference":hist_file,
    }

    print("[6/7] Loading exact held-out Framework M4 transition library ...")
    fw=load_framework_library(results,ctx)
    files["framework_transition_library"]=save_df(fw,outdir/"framework_transition_library")
    errq=framework_error_quantiles(fw)

    print("[7/7] Writing deployment metadata ...")
    horizons=sorted(int(x) for x in fw["horizon_min"].dropna().unique())
    meta={
        "platform_name":"Framework Global Parking Intelligence Platform",
        "serving_semantics":"Framework empirical-state marginal forecast",
        "data_start":emeta["data_start"],"data_end":emeta["data_end"],
        "calendar_source":emeta["calendar_source"],
        "arrival_months":emeta["arrival_months"],"arrival_days_of_week":emeta["arrival_days_of_week"],"arrival_hours":emeta["arrival_hours"],
        "available_horizons":horizons,"framework_error_halfwidth":errq,"files":files,
        "geometry_diagnostics":gdiag,"n_events":emeta["n_exact_history_events"],"n_ext_ids":int(ctx["ext_id"].nunique()),
        "allowed_parking_types":ALLOWED_TYPES,"allowed_parking_geometries":ALLOWED_GEOMETRIES,
        "color_rule":{"green":"occupancy < 5%","orange":"5% <= occupancy <= 50%","red":"occupancy > 50%"},
        "future_forecast_definition":(
            "For future times, actual current occupancy is unknowable. The platform estimates the current occupancy distribution from historical states observed at real arrival_time values, then marginalizes the exact saved Framework M4 transition delta over historically analogous Framework states. No future actual occupancy is used."
        ),
        "historical_validation_definition":(
            "For timestamps inside the observed data interval, actual occupancy is reconstructed exactly from arrival/departure event intervals. If an exact held-out Framework test origin exists, the saved Framework M4 prediction is used directly."
        ),
    }
    (outdir/"serving_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    ui={"months":emeta["arrival_months"],"days_of_week":emeta["arrival_days_of_week"],"hours":emeta["arrival_hours"],"horizons":horizons}
    (outdir/"ui_values.json").write_text(json.dumps(ui,indent=2),encoding="utf-8")
    print(f"[done] Bundle: {outdir.resolve()}")
    print(f"[done] Events: {len(ev):,}; ext_id: {ctx['ext_id'].nunique():,}; polygons: {len(geojson['features']):,}")
    print(f"[done] Historical calendar source: {emeta['calendar_source']}")


if __name__=="__main__":
    main()
