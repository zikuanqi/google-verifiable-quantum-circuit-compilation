# -*- coding: utf-8 -*-
"""Package a submission zip for the competition.

Layout (per the official metric):
    invsqrt.kmx or invsqrt.kmb
    dqi.kmx or dqi.kmb
    dfthcsos/angle_k_0.km[xb] ... angle_k_<K-1>.km[xb]  + submission.npz

Usage:
    python src/package_submission.py <run_dir> -o submissions/run_name.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ALLOWED_TOP = {"invsqrt.kmb", "invsqrt.kmx", "dqi.kmb", "dqi.kmx"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    run = args.run_dir
    invsqrt = [p for p in (run / "invsqrt.kmb", run / "invsqrt.kmx") if p.is_file()]
    dqi = [p for p in (run / "dqi.kmb", run / "dqi.kmx") if p.is_file()]
    dfthcsos = run / "dfthcsos"
    if len(invsqrt) != 1:
        raise SystemExit(f"{run}: need exactly one invsqrt.kmb/.kmx")
    if len(dqi) != 1:
        raise SystemExit(f"{run}: need exactly one dqi.kmb/.kmx")
    if not dfthcsos.is_dir():
        raise SystemExit(f"{run}: missing dfthcsos/ directory")
    angles = sorted(dfthcsos.glob("angle_k_*.kmx")) + sorted(dfthcsos.glob("angle_k_*.kmb"))
    stems = [p.stem for p in angles]
    if not stems or stems != [f"angle_k_{k}" for k in range(len(stems))]:
        raise SystemExit(f"{run}: angle_k_* must be numbered 0..K-1 with no gaps/duplicates")
    if not (dfthcsos / "submission.npz").is_file():
        raise SystemExit(f"{run}: dfthcsos/submission.npz missing")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [invsqrt[0], dqi[0]]:
            zf.write(p, arcname=p.name)
        zf.write(dfthcsos / "submission.npz", arcname="dfthcsos/submission.npz")
        for p in angles:
            zf.write(p, arcname=f"dfthcsos/{p.name}")
    print(f"wrote {args.output}")
    with zipfile.ZipFile(args.output) as zf:
        for info in zf.infolist():
            print(f"  {info.filename}  ({info.file_size:,} bytes)")


if __name__ == "__main__":
    main()
