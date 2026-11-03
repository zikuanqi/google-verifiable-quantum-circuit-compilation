# -*- coding: utf-8 -*-
"""Poll a Kaggle kernel session status and fetch its output log."""
from __future__ import annotations

import sys
import time

from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
from kagglesdk.kaggle_http_client import KaggleHttpClient


def main():
    slug = sys.argv[1]
    _ = sys.argv[2]  # session id not needed by this API
    for i in range(60):
        time.sleep(15)
        with KaggleHttpClient() as client:
            from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
            api = KernelsApiClient(client)
            req = ApiGetKernelSessionStatusRequest()
            req.kernel_slug = slug.split("/")[-1]
            req.user_name = slug.split("/")[0]
            st = api.get_kernel_session_status(req)
        s = str(st)
        print(f"[{i}] {s[:200]}", flush=True)
        if '"done": true' in s.replace(" ", "") or "doneTrue" in s.replace(" ", "").replace('"', "").replace("'", ""):
            break
    with KaggleHttpClient() as client:
        from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
        api = KernelsApiClient(client)
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        req = ApiListKernelSessionOutputRequest()
        req.kernel_slug = slug.split("/")[-1]
        req.user_name = slug.split("/")[0]
        out = api.list_kernel_session_output(req)
        print("OUTPUT:", str(out)[:3000])


if __name__ == "__main__":
    main()
