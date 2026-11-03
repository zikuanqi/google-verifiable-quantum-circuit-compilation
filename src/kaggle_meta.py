# -*- coding: utf-8 -*-
"""Fetch competition metadata from the Kaggle API to learn submission requirements."""

from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest
from kagglesdk.kaggle_http_client import KaggleHttpClient


def main():
    with KaggleHttpClient() as client:
        from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
        api = CompetitionApiClient(client)
        req = ApiGetCompetitionRequest()
        req.competition_name = "google-verifiable-quantum-circuit-compilation"
        comp = api.get_competition(req)
        for f in dir(comp):
            if f.startswith("_"):
                continue
            v = getattr(comp, f)
            if not callable(v):
                print(f, "=", repr(v)[:300])


if __name__ == "__main__":
    main()
