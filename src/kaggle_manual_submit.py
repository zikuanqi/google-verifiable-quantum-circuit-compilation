# -*- coding: utf-8 -*-
"""Resilient submission upload with chunked resumable PUT + retries.

www.googleapis.com is unreachable from this network; storage.googleapis.com
connects but large PUTs get reset. The GCS resumable protocol lets us query the
committed byte offset and resume, so we retry until the whole blob is up.
"""
from __future__ import annotations

import os
import sys
import time

import requests
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiStartSubmissionUploadRequest,
)
from kagglesdk.kaggle_http_client import KaggleHttpClient

COMP = "google-verifiable-quantum-circuit-compilation"


def put_with_resume(url: str, data: bytes, timeout: float = 300.0) -> requests.Response:
    """PUT `data` to a resumable GCS URL, resuming from the committed offset."""
    total = len(data)
    offset = 0
    while True:
        headers = {"Content-Range": f"bytes {offset}-{total - 1}/{total}",
                   "Content-Type": "application/zip"}
        try:
            r = requests.put(url, data=data[offset:], headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            print(f"  PUT error at {offset}/{total}: {e!r}", flush=True)
            r = None
        if r is not None and r.status_code in (200, 201, 204):
            return r
        if r is not None and r.status_code == 308:
            hdr = r.headers.get("Range") or r.headers.get("range")
            if hdr:
                offset = int(hdr.split("-")[-1]) + 1
                print(f"  308 resume at {offset}/{total}", flush=True)
                continue
        # Query the committed offset.
        try:
            q = requests.put(url, data=b"", headers={
                "Content-Range": f"bytes */{total}",
                "Content-Type": "application/zip"}, timeout=120)
            print(f"  status query: {q.status_code}", flush=True)
            if q.status_code == 308:
                hdr = q.headers.get("Range") or q.headers.get("range")
                if hdr:
                    offset = int(hdr.split("-")[-1]) + 1
                    print(f"  committed through {offset - 1}, resuming", flush=True)
        except requests.exceptions.RequestException as e:
            print(f"  query error: {e!r}", flush=True)
        time.sleep(3)


def main():
    zip_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else "submission"
    size = os.path.getsize(zip_path)

    with KaggleHttpClient() as client:
        from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
        api = CompetitionApiClient(client)
        req = ApiStartSubmissionUploadRequest()
        req.competition_name = COMP
        req.file_name = os.path.basename(zip_path)
        req.content_length = size
        req.last_modified_epoch_seconds = int(os.path.getmtime(zip_path))
        print("StartSubmissionUpload ...", flush=True)
        resp = api.start_submission_upload(req)
        print(f"  create_url: {resp.create_url[:110]}...", flush=True)

    url = resp.create_url.replace("www.googleapis.com", "storage.googleapis.com")
    data = open(zip_path, "rb").read()
    t0 = time.time()
    r = put_with_resume(url, data)
    dt = time.time() - t0
    print(f"  PUT done: {r.status_code} in {dt:.1f}s ({size/1024/max(dt,1e-9):.0f} KB/s)", flush=True)
    if r.status_code not in (200, 201, 204):
        raise SystemExit(f"PUT failed: {r.status_code} {r.text[:400]}")

    with KaggleHttpClient() as client:
        from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
        api = CompetitionApiClient(client)
        sreq = ApiCreateSubmissionRequest()
        sreq.competition_name = COMP
        sreq.blob_file_tokens = resp.token
        sreq.submission_description = message
        print("CreateSubmission ...", flush=True)
        created = api.create_submission(sreq)
        print(f"  result: {created.message if hasattr(created, 'message') else created}")


if __name__ == "__main__":
    main()
