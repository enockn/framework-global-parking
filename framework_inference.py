#!/usr/bin/env python3
"""Framework Global Parking Intelligence Platform — inference core.

No alternative ML model is fitted here. Future forecasts are Framework-consistent
empirical marginal forecasts over historically observed parking states.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DAY_ORDER=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
MONTH_NAMES={1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}


def occupancy_band(v: float):
    x=float(np.clip(v,0,1))
    if x<0.05: return "Low (<5%)",[22,163,74,220]
    if x<=0.50: return "Moderate (5–50%)",[245,158,11,220]
    return "High (>50%)",[220,38,38,225]


def _read_bundle_table(bundle: Path, stem_or_name: str) -> pd.DataFrame:
    p=bundle/stem_or_name
    if p.exists():
        if p.suffix.lower()==".parquet": return pd.read_parquet(p)
        return pd.read_csv(p,low_memory=False)
    stem=Path(stem_or_name).stem
    for q in [bundle/(stem+".parquet"),bundle/(stem+".csv.gz"),bundle/(stem+".csv")]:
        if q.exists():
            return pd.read_parquet(q) if q.suffix.lower()==".parquet" else pd.read_csv(q,low_memory=False)
    raise FileNotFoundError(f"Missing bundle table: {stem_or_name}")


def _hour_bin(hour: float) -> float:
    return float(np.floor(float(hour)*4.0)/4.0)


def _month_distance(a,b):
    d=np.abs(np.asarray(a,float)-float(b)); return np.minimum(d,12-d)


def _hour_distance(a,b):
    return np.abs(np.asarray(a,float)-float(b))


@dataclass
class ForecastRequest:
    target_datetime: pd.Timestamp
    horizon_min: int

    @property
    def origin_datetime(self) -> pd.Timestamp:
        return pd.Timestamp(self.target_datetime)-pd.Timedelta(minutes=int(self.horizon_min))


class FrameworkPredictor:
    def __init__(self,bundle_dir: str|Path):
        self.bundle=Path(bundle_dir)
        mp=self.bundle/"serving_metadata.json"
        if not mp.exists(): raise FileNotFoundError(f"Missing serving bundle metadata: {mp}")
        self.meta=json.loads(mp.read_text(encoding="utf-8"))
        files=self.meta.get("files",{})
        self.context=_read_bundle_table(self.bundle,files.get("parking_context","parking_context")) if "parking_context" in files else _read_bundle_table(self.bundle,"parking_context")
        self.context["ext_id"]=pd.to_numeric(self.context["ext_id"],errors="coerce").astype("Int64")
        self.context=self.context.dropna(subset=["ext_id"]).copy(); self.context["ext_id"]=self.context["ext_id"].astype(int)
        for c in ["road_section_uid","road_name","parking_type","type_geom","wayside"]:
            if c in self.context.columns:
                self.context[c]=self.context[c].astype("string").fillna("").astype(str)
        self.ctx_by_ext=self.context.set_index("ext_id",drop=False)

        self.clim_fine=_read_bundle_table(self.bundle,files.get("climatology_ext_calendar","climatology_ext_calendar"))
        self.clim_ext=_read_bundle_table(self.bundle,files.get("climatology_ext","climatology_ext"))
        self.clim_section=_read_bundle_table(self.bundle,files.get("climatology_section_calendar","climatology_section_calendar"))
        self.clim_context=_read_bundle_table(self.bundle,files.get("climatology_context_calendar","climatology_context_calendar"))
        self.clim_global=_read_bundle_table(self.bundle,files.get("climatology_global_calendar","climatology_global_calendar"))
        self.fw=_read_bundle_table(self.bundle,files.get("framework_transition_library","framework_transition_library"))
        self.fw["time"]=pd.to_datetime(self.fw["time"],errors="coerce")
        self.fw["ext_id"]=pd.to_numeric(self.fw["ext_id"],errors="coerce").astype(int)
        self.fw["horizon_min"]=pd.to_numeric(self.fw["horizon_min"],errors="coerce").astype(int)
        for c in ["road_section_uid","road_name","parking_type","type_geom","wayside","day_of_week"]:
            if c in self.fw.columns:
                self.fw[c]=self.fw[c].astype("string").fillna("").astype(str)
        self.data_start=pd.Timestamp(self.meta["data_start"]); self.data_end=pd.Timestamp(self.meta["data_end"])
        self.errq=self.meta.get("framework_error_halfwidth",{})

        # Compact historical event index for exact actual occupancy checks.
        z=np.load(self.bundle/"event_index.npz",allow_pickle=False)
        self.event_ext_ids=z["ext_ids"].astype(int); self.offsets=z["offsets"].astype(np.int64)
        self.arrivals=z["arrivals_ns"].astype(np.int64); self.departures=z["departures_ns"].astype(np.int64)
        self.event_pos={int(e):i for i,e in enumerate(self.event_ext_ids)}
        self.capacity=self.context.set_index("ext_id")["capacity"].astype(float).to_dict()

        # Fast exact held-out Framework lookup.
        self.fw_exact={}
        for i,r in enumerate(self.fw[["ext_id","horizon_min","time"]].itertuples(index=False)):
            if pd.notna(r.time): self.fw_exact[(int(r.ext_id),int(r.horizon_min),int(pd.Timestamp(r.time).value))]=i

        # Candidate-index dictionaries for analogue marginalisation.
        self.fw_ext_groups={k:np.asarray(v,dtype=int) for k,v in self.fw.groupby(["ext_id","horizon_min"],sort=False).indices.items()}
        self.fw_section_groups={k:np.asarray(v,dtype=int) for k,v in self.fw.groupby(["road_section_uid","horizon_min"],sort=False).indices.items()}
        self.fw_context_groups={k:np.asarray(v,dtype=int) for k,v in self.fw.groupby(["parking_type","type_geom","wayside","horizon_min"],sort=False).indices.items()}
        self.fw_horizon_groups={int(k):np.asarray(v,dtype=int) for k,v in self.fw.groupby("horizon_min",sort=False).indices.items()}

    def available_horizons(self)->List[int]:
        return sorted(int(x) for x in self.meta.get("available_horizons",[5,15,30,60]))

    def _climatology(self, rows: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        """Return one climatology row per requested ext_id, preserving row order.

        Pandas merges create a fresh RangeIndex while ``rows`` may retain filtered
        indices from the Streamlit selection table.  Older code mixed the two
        index spaces (``q.loc[miss]``), which raises ``IndexingError: Unalignable
        boolean Series``.  This implementation is deliberately positional:
        every fallback operates on integer row positions and therefore remains
        correct regardless of the caller's DataFrame index.
        """
        q = rows[["ext_id","road_section_uid","parking_type","type_geom","wayside"]].copy().reset_index(drop=True)
        q["month"] = int(origin.month)
        q["day_of_week"] = str(origin.day_name())
        q["hour_bin"] = _hour_bin(origin.hour + origin.minute/60.0)

        # Normalize join-key dtypes. CSV/Parquet inference can otherwise make a
        # UID numeric in one table and textual in another even though they are
        # logically identical.
        for c in ["road_section_uid","parking_type","type_geom","wayside","day_of_week"]:
            if c in q.columns:
                q[c] = q[c].astype("string").fillna("").astype(str)
        q["ext_id"] = pd.to_numeric(q["ext_id"], errors="coerce").astype("Int64")
        q["month"] = pd.to_numeric(q["month"], errors="coerce").astype("Int64")
        q["hour_bin"] = pd.to_numeric(q["hour_bin"], errors="coerce").astype(float)

        cols=["mean_occupancy","median_occupancy","q05","q25","q75","q95","n"]

        def prep_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
            z=df.copy()
            for c in keys:
                if c in {"road_section_uid","parking_type","type_geom","wayside","day_of_week"}:
                    z[c]=z[c].astype("string").fillna("").astype(str)
                elif c in {"ext_id","month"}:
                    z[c]=pd.to_numeric(z[c],errors="coerce").astype("Int64")
                elif c=="hour_bin":
                    z[c]=pd.to_numeric(z[c],errors="coerce").astype(float)
            # The preparation program writes unique grouped keys. Validate here
            # so a corrupted/duplicated serving table fails clearly rather than
            # silently duplicating prediction rows.
            return z

        fine=prep_table(self.clim_fine,["ext_id","month","day_of_week","hour_bin"])
        section=prep_table(self.clim_section,["road_section_uid","month","day_of_week","hour_bin"])
        context=prep_table(self.clim_context,["parking_type","type_geom","wayside","month","day_of_week","hour_bin"])
        ext=prep_table(self.clim_ext,["ext_id"])
        global_cal=prep_table(self.clim_global,["month","day_of_week","hour_bin"])

        a=q.merge(
            fine, on=["ext_id","month","day_of_week","hour_bin"], how="left",
            sort=False, validate="many_to_one"
        ).reset_index(drop=True)
        if len(a) != len(q):
            raise RuntimeError("Climatology merge changed row count; rebuild the serving bundle.")
        source=np.full(len(a),"",dtype=object)
        source[a["mean_occupancy"].notna().to_numpy()]="ext_id + month + weekday + time"

        def fill_from(table: pd.DataFrame, keys: list[str], label: str):
            miss=a["mean_occupancy"].isna().to_numpy()
            pos=np.flatnonzero(miss)
            if len(pos)==0:
                return
            left=q.iloc[pos][keys].copy().reset_index(drop=True)
            b=left.merge(table[keys+cols],on=keys,how="left",sort=False,validate="many_to_one")
            if len(b)!=len(pos):
                raise RuntimeError(f"{label} climatology fallback changed row count; rebuild the serving bundle.")
            for c in cols:
                a.iloc[pos,a.columns.get_loc(c)] = pd.to_numeric(b[c],errors="coerce").to_numpy()
            got=pd.to_numeric(b["mean_occupancy"],errors="coerce").notna().to_numpy()
            source[pos[got]]=label

        fill_from(section,["road_section_uid","month","day_of_week","hour_bin"],"road section + calendar")
        fill_from(context,["parking_type","type_geom","wayside","month","day_of_week","hour_bin"],"parking context + calendar")
        fill_from(ext,["ext_id"],"ext_id historical distribution")

        # Final global calendar fallback: use the closest observed quarter-hour
        # for the same month and weekday. This is positional as well.
        miss=a["mean_occupancy"].isna().to_numpy()
        pos=np.flatnonzero(miss)
        if len(pos):
            gb=global_cal[(global_cal["month"]==int(origin.month)) &
                          (global_cal["day_of_week"].astype(str)==origin.day_name())].copy()
            if len(gb):
                target_bin=_hour_bin(origin.hour+origin.minute/60.0)
                j=int(np.argmin(np.abs(gb["hour_bin"].to_numpy(float)-target_bin)))
                rr=gb.iloc[j]
                for c in cols:
                    a.iloc[pos,a.columns.get_loc(c)] = pd.to_numeric(rr[c],errors="coerce")
                source[pos]="global calendar fallback"

        a["climatology_source"]=source
        for c in cols:
            a[c]=pd.to_numeric(a[c],errors="coerce")
        a["mean_occupancy"]=a["mean_occupancy"].fillna(0.0).clip(0,1)
        a["median_occupancy"]=a["median_occupancy"].fillna(a["mean_occupancy"]).clip(0,1)
        a["n"]=a["n"].fillna(0)
        for c in ["q05","q25","q75","q95"]:
            a[c]=a[c].fillna(a["mean_occupancy"]).clip(0,1)
        a.loc[a["climatology_source"].eq(""),"climatology_source"]="zero fallback (no historical analogue)"
        return a

    def _candidate_indices(self, ctxrow, h:int):
        ext=int(ctxrow.ext_id); sec=str(ctxrow.road_section_uid)
        key=(ext,h)
        if key in self.fw_ext_groups and len(self.fw_ext_groups[key]): return self.fw_ext_groups[key],"ext_id"
        k2=(sec,h)
        if k2 in self.fw_section_groups and len(self.fw_section_groups[k2]): return self.fw_section_groups[k2],"road_section_uid"
        k3=(str(ctxrow.parking_type),str(ctxrow.type_geom),str(ctxrow.wayside),h)
        if k3 in self.fw_context_groups and len(self.fw_context_groups[k3]): return self.fw_context_groups[k3],"parking context"
        return self.fw_horizon_groups.get(h,np.array([],dtype=int)),"global Framework"

    def _framework_delta(self, ctxrow, baseline:float, origin:pd.Timestamp, h:int, k:int=32):
        exact_key=(int(ctxrow.ext_id),int(h),int(origin.value))
        if exact_key in self.fw_exact:
            r=self.fw.iloc[self.fw_exact[exact_key]]
            return float(r["M4"]-r["occ_now"]),"exact held-out Framework",1,float(r["M4"]),float(r["y"]) if "y" in r and np.isfinite(r["y"]) else np.nan
        idx,level=self._candidate_indices(ctxrow,h)
        if not len(idx): return 0.0,"no Framework analogue",0,np.nan,np.nan
        g=self.fw.iloc[idx]
        mo=_month_distance(g["month"].to_numpy(float),origin.month)/6.0
        ho=_hour_distance(g["hour_of_day"].to_numpy(float),origin.hour+origin.minute/60.0)/12.0
        dd=(g["day_of_week"].astype(str).to_numpy()!=origin.day_name()).astype(float)
        oo=np.abs(g["occ_now"].to_numpy(float)-float(baseline))
        score=2.8*oo+0.9*dd+0.65*ho+0.35*mo
        kk=min(max(int(k),1),len(g)); take=np.argpartition(score,kk-1)[:kk]
        s=score[take]; w=np.exp(-s/0.35); w=w/np.sum(w) if np.sum(w)>0 else np.full(kk,1/kk)
        delta=float(np.sum(w*g.iloc[take]["delta_framework"].to_numpy(float)))
        return delta,f"Framework analogue: {level}",kk,np.nan,np.nan

    def actual_occupancy(self, ext_ids:Sequence[int], target_datetime:pd.Timestamp)->pd.DataFrame:
        t=pd.Timestamp(target_datetime)
        out=[]; ns=int(t.value)
        for ext in ext_ids:
            ext=int(ext); pos=self.event_pos.get(ext)
            if pos is None: continue
            lo=int(self.offsets[pos]); hi=int(self.offsets[pos+1])
            arr=self.arrivals[lo:hi]; dep=self.departures[lo:hi]
            count=int(np.searchsorted(arr,ns,side="right")-np.searchsorted(dep,ns,side="right"))
            C=max(float(self.capacity.get(ext,1)),1.0); count=max(0,min(count,int(round(C))))
            out.append((ext,count/C,count))
        return pd.DataFrame(out,columns=["ext_id","actual_occupancy","actual_occupied_spaces"])

    def predict(self, ext_ids:Sequence[int], request:ForecastRequest)->pd.DataFrame:
        ids={int(x) for x in ext_ids}; rows=self.context[self.context["ext_id"].isin(ids)].drop_duplicates("ext_id").copy()
        if rows.empty: return rows.assign(predicted_occupancy=pd.Series(dtype=float))
        target=pd.Timestamp(request.target_datetime); origin=request.origin_datetime; h=int(request.horizon_min)
        clim=self._climatology(rows,origin)
        rows=rows.reset_index(drop=True)
        clim=clim.reset_index(drop=True)
        pred=[]; sources=[]; nanalog=[]; exact_pred=[]; exact_y=[]
        for i,r in enumerate(rows.itertuples(index=False)):
            b=float(clim.loc[i,"mean_occupancy"])
            d,src,n,ep,ey=self._framework_delta(r,b,origin,h)
            pred.append(float(np.clip(b+d,0,1))); sources.append(src); nanalog.append(n); exact_pred.append(ep); exact_y.append(ey)
        out=rows.copy(); out["historical_state_occupancy"]=clim["mean_occupancy"].to_numpy(float)
        out["historical_q05"]=clim["q05"].to_numpy(float); out["historical_q95"]=clim["q95"].to_numpy(float)
        out["historical_state_source"]=clim["climatology_source"].astype(str).to_numpy()
        out["predicted_occupancy"]=np.asarray(pred,float); out["framework_source"]=sources; out["framework_analogues"]=nanalog

        # When exact held-out Framework prediction exists, use it directly instead of the marginal forecast.
        ex=np.isfinite(np.asarray(exact_pred,float))
        if ex.any(): out.loc[ex,"predicted_occupancy"]=np.asarray(exact_pred,float)[ex]

        q=self.errq.get(str(h),{}); fw90=float(q.get("q90",0.02)); fw95=float(q.get("q95",0.04))
        hist_half90=np.maximum(out["historical_state_occupancy"]-out["historical_q05"],out["historical_q95"]-out["historical_state_occupancy"]).to_numpy(float)
        hw90=np.sqrt(hist_half90**2+fw90**2); hw95=np.sqrt((1.15*hist_half90)**2+fw95**2)
        p=out["predicted_occupancy"].to_numpy(float)
        out["PI90_low"]=np.clip(p-hw90,0,1); out["PI90_high"]=np.clip(p+hw90,0,1)
        out["PI95_low"]=np.clip(p-hw95,0,1); out["PI95_high"]=np.clip(p+hw95,0,1)
        C=pd.to_numeric(out["capacity"],errors="coerce").fillna(1).clip(lower=1).to_numpy(float)
        out["expected_occupied_spaces"]=p*C; out["expected_available_spaces"]=(1-p)*C

        # Historical actuals are only available within observed data; future is forecast-only by design.
        historical=(target>=self.data_start) and (target<=self.data_end)
        if historical:
            act=self.actual_occupancy(out["ext_id"].tolist(),target)
            out=out.merge(act,on="ext_id",how="left")
            out["absolute_error"]=np.abs(out["actual_occupancy"]-out["predicted_occupancy"])
            out["forecast_mode"]="Historical validation"
        else:
            out["actual_occupancy"]=np.nan; out["actual_occupied_spaces"]=np.nan; out["absolute_error"]=np.nan
            out["forecast_mode"]="Future forecast only"

        p=out["predicted_occupancy"].to_numpy(float)
        bands,colors=zip(*(occupancy_band(x) for x in p)); out["occupancy_band"]=bands; out["fill_color"]=colors
        out["target_datetime"]=str(target); out["origin_datetime"]=str(origin)
        out["target_month"]=target.month; out["target_day_of_week"]=target.day_name(); out["target_hour_of_day"]=target.hour+target.minute/60.0
        out["origin_month"]=origin.month; out["origin_day_of_week"]=origin.day_name(); out["origin_hour_of_day"]=origin.hour+origin.minute/60.0
        out["forecast_horizon_min"]=h
        return out

    def model_card(self):
        return self.meta.get("future_forecast_definition","")
