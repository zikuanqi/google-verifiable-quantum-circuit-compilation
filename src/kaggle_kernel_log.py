# -*- coding: utf-8 -*-
"""Fetch the latest session log of a kernel and print it fully."""
from __future__ import annotations

import json
import sys

from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from kagglesdk.kaggle_http_client import KaggleHttpClient


def main():
    slug = sys.argv[1]
    with KaggleHttpClient() as client:
        from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
        api = KernelsApiClient(client)
        req = ApiListKernelSessionOutputRequest()
        req.kernel_slug = slug.split("/")[-1]
        req.user_name = slug.split("/")[0]
        out = api.list_kernel_session_output(req)
        log = json.loads(out.log)
        for entry in log:
            print(entry["data"], end="")


if __name__ == "__main__":
    main()
