from __future__ import annotations

import argparse

import pytest

from static_features.download_fire_behavior_assets import (
    _json_url,
    landfire_layers,
    parse_bbox,
    submit_landfire,
    wait_for_landfire,
)


def test_parse_bbox_accepts_wgs84_extent():
    assert parse_bbox("-96.8,34.8,-88.1,41.8") == (-96.8, 34.8, -88.1, 41.8)


@pytest.mark.parametrize("value", ["-96.8,34.8,-88.1", "-88,35,-96,42", "bad"])
def test_parse_bbox_rejects_invalid_extent(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bbox(value)


def test_landfire_layers_requires_explicit_release():
    assert landfire_layers("lf2025") == ("LF2025_FBFM40", "LF2025_CC", "LF2025_CH")
    with pytest.raises(ValueError):
        landfire_layers("latest")


def test_json_url_finds_nested_download_url():
    payload = {"result": {"download": {"fileUrl": "https://example.test/job.zip"}}}
    assert _json_url(payload) == "https://example.test/job.zip"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.ok = True
        self.status_code = 200
        self.text = str(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        if url.endswith("/submit"):
            return FakeResponse({"jobId": "abc-123"})
        return FakeResponse(
            {
                "jobId": "abc-123",
                "status": "Succeeded",
                "result": {"downloadUrl": "https://example.test/result.zip"},
            }
        )


def test_lfps_submit_and_status_are_requestable_without_network():
    session = FakeSession()
    job_id, layer_list = submit_landfire(
        session,
        "analyst@example.org",
        "LF2025",
        (-96.8, 34.8, -88.1, 41.8),
        timeout=5,
    )
    assert job_id == "abc-123"
    assert layer_list == "LF2025_FBFM40;LF2025_CC;LF2025_CH"
    assert wait_for_landfire(session, job_id, timeout=5, poll_seconds=0, max_wait_seconds=1).endswith(
        "/result.zip"
    )
    assert session.calls[0][2]["params"]["Area_of_Interest"] == "-96.8 34.8 -88.1 41.8"
