# -*- coding: utf-8 -*-
"""Submit to the competition via the Python API with verbose progress."""
from __future__ import annotations

import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
path = sys.argv[1]
message = sys.argv[2] if len(sys.argv) > 2 else "submission"

t0 = time.time()
try:
    api.competition_submit(path, message, "google-verifiable-quantum-circuit-compilation")
    print(f"submitted in {time.time() - t0:.0f}s")
except Exception as e:
    print(f"FAILED after {time.time() - t0:.0f}s: {e!r}")
    raise
