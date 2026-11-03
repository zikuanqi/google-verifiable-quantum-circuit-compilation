# -*- coding: utf-8 -*-
"""Create/update a Kaggle script kernel via save_kernel (inline text, no GCS),
then start a session to run it."""
from __future__ import annotations

import sys
import time

from kagglesdk.kernels.types.kernels_api_service import (
    ApiCreateKernelSessionRequest,
    ApiSaveKernelRequest,
)
from kagglesdk.kaggle_http_client import KaggleHttpClient


def main():
    slug = sys.argv[1]          # e.g. "zikuanqi/submit-helper"
    script_path = sys.argv[2]   # local script file
    run = "--run" in sys.argv

    with KaggleHttpClient() as client:
        from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
        api = KernelsApiClient(client)
        req = ApiSaveKernelRequest()
        req.slug = slug
        req.new_title = slug.split("/")[-1]
        req.language = "python"
        req.kernel_type = "script"
        req.is_private = False
        req.enable_internet = True
        req.competition_data_sources = ["google-verifiable-quantum-circuit-compilation"]
        req.session_timeout_seconds = 3600
        req.text = open(script_path, encoding="utf-8").read()
        print("save_kernel ...", flush=True)
        resp = api.save_kernel(req)
        print("  saved:", resp, flush=True)

    if run:
        with KaggleHttpClient() as client:
            from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
            api = KernelsApiClient(client)
            sreq = ApiCreateKernelSessionRequest()
            sreq.slug = slug
            sreq.kernel_type = "script"
            sreq.language = "python"
            sreq.enable_internet = True
            print("create_kernel_session ...", flush=True)
            op = api.create_kernel_session(sreq)
            print("  session:", op, flush=True)


if __name__ == "__main__":
    main()
