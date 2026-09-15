"""Endpoint-level tests for the transect (traversal profile) API."""

from __future__ import annotations

from fractions import Fraction


def poly(exterior, holes=None):
    return {"exterior": exterior, "holes": holes or []}


def post(client, a, b, path):
    return client.post(
        "/api/v1/transect", json={"a": a, "b": b, "path": path}
    )


def frac(pair):
    return Fraction(pair[0], pair[1])


def flattened_intervals(data):
    """(segment_index, start, end, a, b) for every interval in order."""
    out = []
    for seg in data["segments"]:
        for iv in seg["intervals"]:
            out.append(
                (seg["segment"], frac(iv["start"]), frac(iv["end"]),
                 iv["a"], iv["b"])
            )
    return out


def assert_coverage(data, n_segments):
    assert [s["segment"] for s in data["segments"]] == list(range(n_segments))
    for seg in data["segments"]:
        ivs = seg["intervals"]
        assert ivs, "every segment must have at least one interval"
        assert frac(ivs[0]["start"]) == 0
        assert frac(ivs[-1]["end"]) == 1
        for lo, hi in zip(ivs, ivs[1:]):
            assert frac(lo["end"]) == frac(hi["start"])
        for iv in ivs:
            assert frac(iv["start"]) < frac(iv["end"])
        # Parameters are integer fraction pairs, never floating point.
        for iv in ivs:
            for key in ("start", "end"):
                assert all(isinstance(x, int) for x in iv[key])
        for ct in seg["contacts"]:
            assert all(isinstance(x, int) for x in ct["at"])
            assert ct["at_a"] in {"outside", "inside", "boundary"}
            assert ct["at_b"] in {"outside", "inside", "boundary"}


def test_path_through_exterior_hole_and_two_groups(client):
    # Horizontal line y=5 from x=-2 to x=12 crosses, in order:
    # A exterior (x=0), B exterior (x=2), A hole (x=4,x=6),
    # B exterior (x=8), A exterior (x=10); dx = 14.
    payload_a = [poly([(0, 0), (10, 0), (10, 10), (0, 10)],
                      [[(4, 4), (6, 4), (6, 6), (4, 6)]])]
    payload_b = [poly([(2, 2), (8, 2), (8, 8), (2, 8)])]
    resp = post(client, payload_a, payload_b, [[-2, 5], [12, 5]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert_coverage(data, 1)

    seq = [(s, e, a, b) for _, s, e, a, b in flattened_intervals(data)]
    assert seq == [
        (Fraction(0), Fraction(1, 7), "outside", "outside"),
        (Fraction(1, 7), Fraction(2, 7), "inside", "outside"),
        (Fraction(2, 7), Fraction(3, 7), "inside", "inside"),
        (Fraction(3, 7), Fraction(4, 7), "outside", "inside"),  # through the hole
        (Fraction(4, 7), Fraction(5, 7), "inside", "inside"),
        (Fraction(5, 7), Fraction(6, 7), "inside", "outside"),
        (Fraction(6, 7), Fraction(1), "outside", "outside"),
    ]

    contacts = data["segments"][0]["contacts"]
    assert [frac(c["at"]) for c in contacts] == [
        Fraction(1, 7), Fraction(2, 7), Fraction(3, 7),
        Fraction(4, 7), Fraction(5, 7), Fraction(6, 7),
    ]
    # Crossing into A through its exterior.
    assert (contacts[0]["before_a"], contacts[0]["at_a"],
            contacts[0]["after_a"]) == ("outside", "boundary", "inside")
    assert contacts[0]["at_b"] == "outside"
    # Entering the hole: A goes inside -> boundary -> outside.
    assert (contacts[2]["before_a"], contacts[2]["at_a"],
            contacts[2]["after_a"]) == ("inside", "boundary", "outside")
    assert contacts[2]["at_b"] == "inside"
    # Leaving the hole again.
    assert (contacts[3]["before_a"], contacts[3]["at_a"],
            contacts[3]["after_a"]) == ("outside", "boundary", "inside")


def test_multi_segment_path_covers_each_segment(client):
    a = [poly([(0, 0), (4, 0), (4, 4), (0, 4)])]
    resp = post(client, a, [], [[-2, 2], [2, 2], [2, -2]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert_coverage(data, 2)
    seg0, seg1 = data["segments"]
    # Each segment crosses x=2 / y=2 respectively at its midpoint (t=1/2).
    assert [(frac(i["start"]), frac(i["end"]), i["a"])
            for i in seg0["intervals"]] == [
        (Fraction(0), Fraction(1, 2), "outside"),
        (Fraction(1, 2), Fraction(1), "inside"),
    ]
    assert [(frac(i["start"]), frac(i["end"]), i["a"])
            for i in seg1["intervals"]] == [
        (Fraction(0), Fraction(1, 2), "inside"),
        (Fraction(1, 2), Fraction(1), "outside"),
    ]
    # The shared fold vertex on the boundary is an isolated contact for both
    # original segments, each with the neighbouring side filled in.
    c0 = seg0["contacts"][0]
    c1 = seg1["contacts"][0]
    assert frac(c0["at"]) == Fraction(1, 2) and frac(c1["at"]) == Fraction(1, 2)
    assert (c0["before_a"], c0["at_a"], c0["after_a"]) == (
        "outside", "boundary", "inside")
    assert (c1["before_a"], c1["at_a"], c1["after_a"]) == (
        "inside", "boundary", "outside")


def test_vertex_tangency_is_isolated_contact(client):
    # The polyline touches the single vertex (2,1) of the square and stays
    # outside on both sides.
    a = [poly([(2, 1), (6, 1), (6, 5), (2, 5)])]
    resp = post(client, a, [], [[0, 3], [2, 1], [0, -1]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert_coverage(data, 2)
    for seg in data["segments"]:
        assert len(seg["intervals"]) == 1
        iv = seg["intervals"][0]
        assert (frac(iv["start"]), frac(iv["end"]), iv["a"]) == (
            Fraction(0), Fraction(1), "outside")
        assert len(seg["contacts"]) == 1
        ct = seg["contacts"][0]
        assert (ct["before_a"], ct["at_a"], ct["after_a"]) == (
            "outside", "boundary", "outside")
    # The contact sits at t=1 of segment 0 and t=0 of segment 1.
    assert frac(data["segments"][0]["contacts"][0]["at"]) == 1
    assert frac(data["segments"][1]["contacts"][0]["at"]) == 0


def test_path_running_along_collinear_boundary(client):
    # (0,1)->(4,1) follows the bottom edge of the square from x=2 (t=1/2),
    # then (4,1)->(4,3) leaves into the interior: a boundary RUN, not a point.
    a = [poly([(2, 1), (6, 1), (6, 5), (2, 5)])]
    resp = post(client, a, [], [[0, 1], [4, 1], [4, 3]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert_coverage(data, 2)
    seg0, seg1 = data["segments"]
    assert [(frac(i["start"]), frac(i["end"]), i["a"])
            for i in seg0["intervals"]] == [
        (Fraction(0), Fraction(1, 2), "outside"),
        (Fraction(1, 2), Fraction(1), "boundary"),
    ]
    assert [(frac(i["start"]), frac(i["end"]), i["a"])
            for i in seg1["intervals"]] == [
        (Fraction(0), Fraction(1), "inside"),
    ]
    # No isolated contact: the boundary contact spans a whole interval and
    # continues through the fold vertex.
    assert seg0["contacts"] == []
    assert seg1["contacts"] == []


def test_fractional_intersections_forward_and_reverse(client):
    # Sloped line through a square with a hole and a second group produces
    # only fractional cut parameters.
    a = [poly([(0, 0), (10, 0), (10, 10), (0, 10)],
              [[(4, 4), (6, 4), (6, 6), (4, 6)]])]
    b = [poly([(2, -2), (8, -2), (8, 4), (2, 4)])]
    forward = [[-1, 1], [11, 7]]

    resp = post(client, a, b, forward)
    assert resp.status_code == 200, resp.text
    fwd = resp.json()
    assert_coverage(fwd, 1)
    cuts = [frac(i["start"]) for i in fwd["segments"][0]["intervals"]][1:]
    assert cuts == [Fraction(1, 12), Fraction(1, 4), Fraction(1, 2),
                    Fraction(7, 12), Fraction(11, 12)]
    # Every cut is a non-trivial fraction; nothing is serialised as a float.
    assert all(t.denominator != 1 for t in cuts)

    resp_rev = post(client, a, b, list(reversed(forward)))
    assert resp_rev.status_code == 200, resp_rev.text
    rev = resp_rev.json()
    assert_coverage(rev, 1)

    fi = fwd["segments"][0]["intervals"]
    ri = rev["segments"][0]["intervals"]
    assert len(fi) == len(ri)
    # Reversing the whole polyline mirrors every interval t -> 1 - t and
    # preserves its group relations.
    for fwd_iv, rev_iv in zip(fi, reversed(ri)):
        assert frac(rev_iv["start"]) == 1 - frac(fwd_iv["end"])
        assert frac(rev_iv["end"]) == 1 - frac(fwd_iv["start"])
        assert (rev_iv["a"], rev_iv["b"]) == (fwd_iv["a"], fwd_iv["b"])
    # Events correspond one-to-one and before/after swap in every contact.
    fc = {frac(c["at"]): c for c in fwd["segments"][0]["contacts"]}
    rc = {frac(c["at"]): c for c in rev["segments"][0]["contacts"]}
    assert set(fc) == {1 - t for t in rc}
    for t, c in fc.items():
        d = rc[1 - t]
        assert (d["before_a"], d["at_a"], d["after_a"]) == (
            c["after_a"], c["at_a"], c["before_a"])
        assert (d["before_b"], d["at_b"], d["after_b"]) == (
            c["after_b"], c["at_b"], c["before_b"])


def test_empty_groups_profile_everything_outside(client):
    resp = post(client, [], [], [[0, 0], [3, 3], [3, 0]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert_coverage(data, 2)
    for seg in data["segments"]:
        assert len(seg["intervals"]) == 1
        iv = seg["intervals"][0]
        assert (iv["start"], iv["end"], iv["a"], iv["b"]) == (
            [0, 1], [1, 1], "outside", "outside")
        assert seg["contacts"] == []


def test_endpoint_contact_triples_have_missing_sides(client):
    # The path starts on the left edge heading in and ends on the right edge:
    # contacts at the very ends of the whole polyline have no outside side.
    a = [poly([(0, 0), (4, 0), (4, 4), (0, 4)])]
    resp = post(client, a, [], [[0, 2], [2, 2], [4, 2]])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    seg0, seg1 = data["segments"]

    first = {(frac(c["at"]), c["at_a"]) for c in seg0["contacts"]}
    assert (Fraction(0), "boundary") in first
    first_contact = next(c for c in seg0["contacts"] if frac(c["at"]) == 0)
    assert first_contact["before_a"] is None
    assert (first_contact["at_a"], first_contact["after_a"]) == (
        "boundary", "inside")

    last_contact = next(c for c in seg1["contacts"] if frac(c["at"]) == 1)
    assert last_contact["after_a"] is None
    assert (last_contact["before_a"], last_contact["at_a"]) == (
        "inside", "boundary")


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------

def assert_422(resp):
    assert resp.status_code == 422, resp.text
    err = resp.json()["error"]
    assert err["code"] in {"invalid_request", "invalid_geometry"}
    return err


def test_duplicate_consecutive_path_points_are_422_at_position(client):
    resp = post(
        client,
        [poly([(0, 0), (4, 0), (4, 4), (0, 4)])], [],
        [[0, 2], [2, 2], [2, 2], [5, 2]],
    )
    err = assert_422(resp)
    assert err["code"] == "invalid_request"
    locs = [tuple(d["loc"]) for d in err["details"]]
    # Schema-error loc parts are stringified by the structured error handler.
    assert ("body", "path", "2") in locs


def test_first_duplicate_pair_also_rejected(client):
    resp = post(client, [], [], [[1, 1], [1, 1], [2, 2]])
    err = assert_422(resp)
    assert ("body", "path", "1") in [tuple(d["loc"]) for d in err["details"]]


def test_path_with_fewer_than_two_points_422(client):
    for bad_path in ([[0, 0]], []):
        resp = post(client, [], [], bad_path)
        assert resp.status_code == 422, bad_path
        assert resp.json()["error"]["code"] == "invalid_request"


def test_path_coordinate_type_and_range_422(client):
    for bad in ([0, 1.5], [True, 1], [0, "1"], [1_000_001, 0], [1]):
        resp = post(client, [], [], [[0, 0], bad])
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "invalid_request"
        loc = tuple(resp.json()["error"]["details"][0]["loc"])
        assert loc[:3] == ("body", "path", "1")


def test_malformed_path_container_422(client):
    for bad in (None, 5, "x"):
        resp = client.post(
            "/api/v1/transect",
            json={"a": [], "b": [], "path": bad},
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "invalid_request"


def test_bad_polygon_keeps_existing_error_location(client):
    bowtie = [(0, 0), (4, 4), (4, 0), (0, 4)]
    resp = post(
        client,
        [poly([(0, 0), (1, 0), (1, 1), (0, 1)]), poly(bowtie)],
        [], [[-1, -1], [5, 5]],
    )
    err = assert_422(resp)
    assert err["code"] == "invalid_geometry"
    assert tuple(err["details"][0]["loc"][:2]) == ("a", 1)


def test_hole_not_strictly_inside_rejected(client):
    resp = post(
        client,
        [poly([(0, 0), (4, 0), (4, 4), (0, 4)],
              [[(0, 0), (1, 0), (1, 1), (0, 1)]])],
        [], [[0, 0], [2, 2]],
    )
    err = assert_422(resp)
    assert err["code"] == "invalid_geometry"
    assert tuple(err["details"][0]["loc"][:3]) == ("a", 0, "holes")


def test_overlap_endpoint_still_works_after_rejected_transect(client):
    bad = post(
        client, [poly([(0, 0), (4, 0), (4, 4), (0, 4)])], [],
        [[0, 0], [0, 0]],
    )
    assert bad.status_code == 422

    resp = client.post(
        "/api/v1/overlap",
        json={
            "a": [poly([(0, 0), (4, 0), (4, 4), (0, 4)])],
            "b": [poly([(2, 0), (6, 0), (6, 4), (2, 4)])],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["area_sq_mm"] == [8, 1]
    assert body["decimal"] == "8.000"
    assert body["rounding"] == "half-up"
    assert body["units"] == "mm^2"
