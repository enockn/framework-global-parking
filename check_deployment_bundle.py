#!/usr/bin/env python3
"""Pre-push validation for the Framework Streamlit deployment repository."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "framework_serving"
REQUIRED = ["serving_metadata.json", "parking_spaces.geojson", "event_index.npz"]

def human(n: int) -> str:
    units=["B","KB","MB","GB"]
    x=float(n)
    for u in units:
        if x < 1024 or u==units[-1]:
            return f"{x:.1f} {u}"
        x/=1024

def main() -> int:
    print(f"Repository: {ROOT}")
    missing=[x for x in REQUIRED if not (BUNDLE/x).exists()]
    if missing:
        print("ERROR: serving bundle is incomplete. Missing:", ", ".join(missing))
        return 2
    meta=json.loads((BUNDLE/"serving_metadata.json").read_text(encoding="utf-8"))
    print("Framework horizons:", meta.get("available_horizons"))
    print("Observed interval:", meta.get("data_start"), "to", meta.get("data_end"))
    total=0; too_large=[]
    print("\nServing artifacts:")
    for p in sorted(BUNDLE.iterdir()):
        if p.is_file():
            n=p.stat().st_size; total+=n
            mark=""
            if n >= 100*1024*1024:
                too_large.append(p.name); mark="  [Git LFS REQUIRED]"
            print(f"  {p.name:48s} {human(n):>10s}{mark}")
    print("\nTotal serving bundle:", human(total))
    if too_large:
        print("\nLarge files detected. Ensure Git LFS is installed before git add:")
        print("  git lfs install")
        print("Tracked patterns are already declared in .gitattributes.")
    print("\nOK: deployment bundle is structurally ready.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
