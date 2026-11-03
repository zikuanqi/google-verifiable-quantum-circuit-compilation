# -*- coding: utf-8 -*-
"""Fetch competition pages (overview/evaluation/data) via the Kaggle API."""
import json

from kagglesdk.competitions.types.competition_api_service import ApiListCompetitionPagesRequest
from kagglesdk.kaggle_http_client import KaggleHttpClient


def main():
    with KaggleHttpClient() as client:
        from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
        api = CompetitionApiClient(client)
        req = ApiListCompetitionPagesRequest()
        req.competition_name = "google-verifiable-quantum-circuit-compilation"
        pages = api.list_competition_pages(req)
        for p in getattr(pages, "pages", []) or []:
            print("=" * 60)
            print("name:", getattr(p, "name", ""), "| endpoint:", getattr(p, "endpoint_path", ""),
                  "| mime:", getattr(p, "mime_type", ""))
            content = getattr(p, "content", "")
            print((content or "")[:8000])


if __name__ == "__main__":
    main()
