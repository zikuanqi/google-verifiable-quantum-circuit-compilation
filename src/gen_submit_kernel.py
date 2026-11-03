# -*- coding: utf-8 -*-
"""Generate the submission kernel: embeds builder sources (base64) and data
blobs, builds each benchmark's circuit inside Kaggle (kickmix available), packs
the zip and submits.

Config (JSON) selects what to build:
    {
      "message": "...",
      "sources": ["b1_v2_model.py", "b1_fit.py", "b1_v2_build.py"],
      "blobs": {"b3_cand.npz": "results/b3_cand.npz"},
      "benchmarks": {"invsqrt": "b1_v2", "dqi": "sample", "dfthcsos": "sample"}
    }

Usage: python src/gen_submit_kernel.py --config cfg.json
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADER = '''"""Kaggle kernel: build circuits, package submission.zip, submit."""
import base64
import gzip
import os
import shutil
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "google-verifiable-quantum-circuit-compilation"
MESSAGE = {message!r}

SOURCES = {sources!r}
for name, b64 in SOURCES.items():
    with open(name, "w") as f:
        f.write(base64.b64decode(b64).decode("utf-8"))

BLOBS = {blobs!r}
for name, b64 in BLOBS.items():
    with open(name, "wb") as f:
        f.write(gzip.decompress(base64.b64decode(b64)))

comp = None
for root, dirs, files in os.walk("/kaggle/input/competitions"):
    if os.path.basename(root) == "google-verifiable-quantum-circuit-compilation":
        comp = root
        break
assert comp, "competition data not found"
print("competition data at:", comp)
sample = os.path.join(comp, "sample_submission")

import sys

sys.path.insert(0, os.path.join(comp, "kickmix"))
sys.path.insert(0, "/kaggle/working")

RUN = "/kaggle/working/run"
os.makedirs(os.path.join(RUN, "dfthcsos"), exist_ok=True)

invsqrt_mode = {invsqrt_mode!r}
if invsqrt_mode == "b1_v2":
    import b1_v2_build
    import b1_v2_model
    prec = b1_v2_model.V2Precision(K=32, degree=2, coef_bits=17, delta_bits=12,
                                   square_bits=28, product_bits=28, result_bits=28)
    circuit = b1_v2_build.build_v2_circuit(10, prec)
    circuit.to_file(os.path.join(RUN, "invsqrt.kmb"), format="kmb")
    st = circuit.operation_statistics()
    print(f"invsqrt: {{st.num_toffoli_like}} Toffolis, {{circuit.num_qubits}} qubits")
elif invsqrt_mode == "b1_v3":
    import b1v3_build
    import b1v3_model as v3m
    cfg = v3m.V3Cfg(f2={f2}, f1={f1}, fyr={fyr}, gy={gy}, drop2={drop2})
    print("fitting coefficients ...")
    tabs, _ = v3m.fit_all(cfg)
    lay = v3m.make_layout(cfg, tabs)
    circuit = b1v3_build.build_v3_circuit(cfg, tabs, lay)
    circuit.to_file(os.path.join(RUN, "invsqrt.kmb"), format="kmb")
    st = circuit.operation_statistics()
    print(f"invsqrt: {{st.num_toffoli_like}} Toffolis, {{circuit.num_qubits}} qubits")
else:
    shutil.copy(os.path.join(sample, "invsqrt.kmb"), os.path.join(RUN, "invsqrt.kmb"))
    print("invsqrt: sample")

dqi_mode = {dqi_mode!r}
if dqi_mode == "b2":
    import b2_build
    print("building dqi decoder ...")
    circuit = b2_build.build_circuit()
    circuit.to_file(os.path.join(RUN, "dqi.kmb"), format="kmb")
    st = circuit.operation_statistics()
    print(f"dqi: {{st.num_toffoli_like}} Toffolis, {{circuit.num_qubits}} qubits")
else:
    shutil.copy(os.path.join(sample, "dqi.kmb"), os.path.join(RUN, "dqi.kmb"))
    print("dqi: sample")

dfthcsos_mode = {dfthcsos_mode!r}
if dfthcsos_mode == "blob":
    import numpy as np
    import b3_angle_build
    fact = np.load("b3_cand.npz", allow_pickle=False)
    np.savez(os.path.join(RUN, "dfthcsos", "submission.npz"),
             **{{k: fact[k] for k in ("U", "W", "H2_I", "H1_I", "Hsym_lower", "tree_edges")}})
    b3_angle_build.build_family(fact, os.path.join(RUN, "dfthcsos"))
    print("dfthcsos: built from blob")
else:
    for f in os.listdir(os.path.join(sample, "dfthcsos")):
        shutil.copy(os.path.join(sample, "dfthcsos", f),
                    os.path.join(RUN, "dfthcsos", f))
    print("dfthcsos: sample")

zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(RUN):
        for f in files:
            full = os.path.join(root, f)
            zf.write(full, arcname=os.path.relpath(full, RUN))
print("zip entries:", len(zipfile.ZipFile(zip_path).namelist()))

api = KaggleApi()
api.authenticate()
api.competition_submit(zip_path, MESSAGE, COMP)
print("SUBMITTED OK")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "kernels" / "submit_helper" / "submit_helper.py")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    sources = {}
    for name in cfg.get("sources", []):
        sources[name] = base64.b64encode((ROOT / "src" / name).read_bytes()).decode("ascii")
    blobs = {}
    for name, path in cfg.get("blobs", {}).items():
        data = (ROOT / path).read_bytes()
        blobs[name] = base64.b64encode(gzip.compress(data, 9)).decode("ascii")

    bench = cfg.get("benchmarks", {})
    text = HEADER.format(
        message=cfg.get("message", "submission"),
        sources=sources,
        blobs=blobs,
        invsqrt_mode=bench.get("invsqrt", "sample"),
        dqi_mode=bench.get("dqi", "sample"),
        dfthcsos_mode=bench.get("dfthcsos", "sample"),
        f2=cfg.get("b1_v3", {}).get("f2", 8),
        f1=cfg.get("b1_v3", {}).get("f1", 19),
        fyr=cfg.get("b1_v3", {}).get("fyr", 27),
        gy=cfg.get("b1_v3", {}).get("gy", 1),
        drop2=tuple(cfg.get("b1_v3", {}).get("drop2", [5])),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({len(text):,} chars; {len(sources)} sources, {len(blobs)} blobs)")


if __name__ == "__main__":
    main()
