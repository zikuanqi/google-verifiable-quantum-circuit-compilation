# -*- coding: utf-8 -*-
"""Cancel any running session of a kernel, then start a fresh one."""
from __future__ import annotations

import sys

from kagglesdk.kernels.types.kernels_api_service import (
    ApiCancelKernelSessionRequest,
    ApiCreateKernelSessionRequest,
)
from kagglesdk.kaggle_http_client import KaggleHttpClient


def main():
    slug = sys.argv[1]
    session_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with KaggleHttpClient() as client:
        from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
        api = KernelsApiClient(client)
        req = ApiCancelKernelSessionRequest()
        req.kernel_session_id = session_id
        print("cancel_kernel_session ...", flush=True)
        try:
            out = api.cancel_kernel_session(req)
            print("  cancel:", out, flush=True)
        except Exception as e:
            print("  cancel error (ignored):", str(e)[:200], flush=True)
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
