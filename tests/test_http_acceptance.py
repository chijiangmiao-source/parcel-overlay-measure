"""End-to-end HTTP acceptance tests run by the ``verify`` Compose service.

The tests target a live server given by ``EXACT_AREA_BASE_URL`` (e.g.
``http://api:8000`` inside Compose).  When the variable is not set the whole
module is skipped, so local ``pytest`` runs do not require a running server.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

BASE_URL = os.environ.get("EXACT_AREA_BASE_URL")

pytestmark = pytest.mark.skipif(
    BASE_URL is None,
    reason="EXACT_AREA_BASE_URL not set; live API acceptance tests skipped",
)


def _post(path: str, payload: dict):
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _get(path: str):
    with urllib.request.urlopen(BASE_URL.rstrip() + path, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def poly(exterior, holes=None):
    return {"exterior": exterior, "holes": holes or []}


def test_health_live():
    status, data = _get("/health")
    assert status == 200
    assert data == {"status": "ok"}


def test_fractional_overlap_live():
    payload = {
        "a": [poly([(0, 0), (3, 0), (0, 3)])],
        "b": [poly([(1, 1), (4, 1), (1, 4)])],
    }
    status, data = _post("/api/v1/overlap", payload)
    assert status == 200, data
    assert data["area_sq_mm"] == [1, 2]
    assert data["numerator"] == 1
    assert data["denominator"] == 2
    assert data["decimal"] == "0.500"
    assert data["rounding"] == "half-up"


def test_holes_and_edge_contact_live():
    payload = {
        "a": [poly([(0, 0), (4, 0), (4, 4), (0, 4)],
                   [[(1, 1), (3, 1), (3, 3), (1, 3)]])],
        "b": [poly([(0, 0), (4, 0), (4, 4), (0, 4)])],
    }
    status, data = _post("/api/v1/overlap", payload)
    assert status == 200
    assert data["area_sq_mm"] == [12, 1]
    assert data["decimal"] == "12.000"

    payload = {
        "a": [poly([(0, 0), (1, 0), (1, 1), (0, 1)])],
        "b": [poly([(1, 0), (2, 0), (2, 1), (1, 1)])],
    }
    status, data = _post("/api/v1/overlap", payload)
    assert status == 200
    assert data["area_sq_mm"] == [0, 1]
    assert data["decimal"] == "0.000"


def test_half_up_rounding_live():
    # A unit square and a long, thin triangle whose upper edge runs through
    # (0, 0) and (1000, 1) overlap in a wedge of area 1/2000 = 0.0005, which
    # rounds half-up to 0.001 (Python's built-in round() would not).
    payload = {
        "a": [poly([(0, 0), (1, 0), (1, 1), (0, 1)])],
        "b": [poly([(0, 0), (1000, 1), (0, -1)])],
    }
    status, data = _post("/api/v1/overlap", payload)
    assert status == 200, data
    assert data["area_sq_mm"] == [1, 2000]
    assert data["decimal"] == "0.001"


def test_invalid_geometry_is_structured_422_live():
    payload = {
        "a": [poly([(0, 0), (4, 4), (4, 0), (0, 4)])],  # bowtie
        "b": [poly([(0, 0), (1, 0), (1, 1), (0, 1)])],
    }
    status, data = _post("/api/v1/overlap", payload)
    assert status == 422
    assert data["error"]["code"] == "invalid_geometry"
    assert isinstance(data["error"]["message"], str) and data["error"]["message"]
    assert "details" in data["error"]


def test_schema_violation_is_structured_422_live():
    status, data = _post("/api/v1/overlap", {"a": []})
    assert status == 422
    assert data["error"]["code"] == "invalid_request"
    assert data["error"]["details"]


# ---------------------------------------------------------------------------
# Transect endpoint
# ---------------------------------------------------------------------------

def transect(a, b, path):
    return _post("/api/v1/transect", {"a": a, "b": b, "path": path})


def test_transect_through_exterior_and_hole_two_groups_live():
    a = [poly([(0, 0), (10, 0), (10, 10), (0, 10)],
              [[(4, 4), (6, 4), (6, 6), (4, 6)]])]
    b = [poly([(2, 2), (8, 2), (8, 8), (2, 8)])]
    status, data = transect(a, b, [[-2, 5], [12, 5]])
    assert status == 200, data
    seg = data["segments"][0]
    assert seg["segment"] == 0
    assert [iv["a"] for iv in seg["intervals"]] == [
        "outside", "inside", "inside", "outside",
        "inside", "inside", "outside",
    ]
    assert [iv["b"] for iv in seg["intervals"]] == [
        "outside", "outside", "inside", "inside",
        "inside", "outside", "outside",
    ]
    # Exact seventh-parameters, never floats.
    assert [[iv["start"], iv["end"]] for iv in seg["intervals"]] == [
        [[0, 1], [1, 7]], [[1, 7], [2, 7]], [[2, 7], [3, 7]],
        [[3, 7], [4, 7]], [[4, 7], [5, 7]], [[5, 7], [6, 7]],
        [[6, 7], [1, 1]],
    ]
    # Full coverage of parameters 0..1 and ordered, non-overlapping cuts.
    assert seg["intervals"][0]["start"] == [0, 1]
    assert seg["intervals"][-1]["end"] == [1, 1]
    for lo, hi in zip(seg["intervals"], seg["intervals"][1:]):
        assert lo["end"] == hi["start"]
    # Hole entry contact records inside -> boundary -> outside for A.
    hole_contact = next(
        c for c in seg["contacts"] if c["at"] == [3, 7]
    )
    assert (hole_contact["before_a"], hole_contact["at_a"],
            hole_contact["after_a"]) == ("inside", "boundary", "outside")
    assert hole_contact["at_b"] == "inside"


def test_transect_vertex_tangency_and_boundary_run_live():
    a = [poly([(2, 1), (6, 1), (6, 5), (2, 5)])]

    # Tangent at vertex (2,1): outside on both sides, isolated contacts only.
    status, data = transect(a, [], [[0, 3], [2, 1], [0, -1]])
    assert status == 200, data
    for seg in data["segments"]:
        assert len(seg["intervals"]) == 1
        assert seg["intervals"][0]["a"] == "outside"
        ct = seg["contacts"][0]
        assert (ct["before_a"], ct["at_a"], ct["after_a"]) == (
            "outside", "boundary", "outside")

    # Running along the collinear bottom edge: a boundary INTERVAL, and no
    # isolated contact at either end of that run.
    status, data = transect(a, [], [[0, 1], [4, 1], [4, 3]])
    assert status == 200, data
    seg0, seg1 = data["segments"]
    assert [[iv["start"], iv["end"], iv["a"]]
            for iv in seg0["intervals"]] == [
        [[0, 1], [1, 2], "outside"],
        [[1, 2], [1, 1], "boundary"],
    ]
    assert [[iv["start"], iv["end"], iv["a"]]
            for iv in seg1["intervals"]] == [[[0, 1], [1, 1], "inside"]]
    assert seg0["contacts"] == [] and seg1["contacts"] == []


def test_transect_fractional_path_reverses_exactly_live():
    a = [poly([(0, 0), (10, 0), (10, 10), (0, 10)],
              [[(4, 4), (6, 4), (6, 6), (4, 6)]])]
    b = [poly([(2, -2), (8, -2), (8, 4), (2, 4)])]
    forward_path = [[-1, 1], [11, 7]]

    status, fwd = transect(a, b, forward_path)
    assert status == 200, fwd
    status, rev = transect(a, b, list(reversed(forward_path)))
    assert status == 200, rev

    fi = fwd["segments"][0]["intervals"]
    ri = rev["segments"][0]["intervals"]
    assert len(fi) == len(ri)
    for f, r in zip(fi, reversed(ri)):
        # t -> 1 - t mirror under reversal; fraction pairs stay integers.
        assert isinstance(f["start"][0], int) and isinstance(r["start"][1], int)
        assert [r["start"][0], r["start"][1]] == [
            f["end"][1] - f["end"][0], f["end"][1]]
        assert [r["end"][0], r["end"][1]] == [
            f["start"][1] - f["start"][0], f["start"][1]]
        assert (r["a"], r["b"]) == (f["a"], f["b"])

    def contacts_by_at(seg):
        return {tuple(c["at"]): c for c in seg["contacts"]}

    fc, rc = (contacts_by_at(fwd["segments"][0]),
              contacts_by_at(rev["segments"][0]))
    # Every forward event maps one-to-one to a reversed event at 1 - t with
    # before/after swapped.
    for (num, den), c in fc.items():
        d = rc[(den - num, den)]
        assert (d["before_a"], d["at_a"], d["after_a"]) == (
            c["after_a"], c["at_a"], c["before_a"])
        assert (d["before_b"], d["at_b"], d["after_b"]) == (
            c["after_b"], c["at_b"], c["before_b"])
    assert set(rc) == {(den - num, den) for num, den in fc}


def test_transect_invalid_path_rejected_and_overlap_still_works_live():
    a = [poly([(0, 0), (4, 0), (4, 4), (0, 4)])]
    status, data = transect(a, [], [[0, 2], [2, 2], [2, 2]])
    assert status == 422
    assert data["error"]["code"] == "invalid_request"
    locs = [tuple(d["loc"]) for d in data["error"]["details"]]
    assert ("body", "path", "2") in locs

    status, data = transect(a, [], [[0, 0]])
    assert status == 422
    assert data["error"]["code"] == "invalid_request"

    # The existing overlap interface is unaffected.
    status, data = _post(
        "/api/v1/overlap",
        {"a": a, "b": [poly([(2, 0), (6, 0), (6, 4), (2, 4)])]},
    )
    assert status == 200, data
    assert data["area_sq_mm"] == [8, 1]
    assert data["decimal"] == "8.000"
