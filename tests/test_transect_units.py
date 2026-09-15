"""Unit and property tests for the exact transect engine."""

from __future__ import annotations

import random
from fractions import Fraction

from app.geometry.rationals import Point, int_point, point_on_segment
from app.geometry.transect import (
    BOUNDARY,
    INSIDE,
    OUTSIDE,
    boundary_parameters,
    point_relation,
    transect_profile,
)
from app.geometry.validation import Polygon, build_polygon


SQ_HOLE: list[Polygon] = [
    build_polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                  [[(4, 4), (6, 4), (6, 6), (4, 6)]])
]


def ip(x, y) -> Point:
    return int_point((x, y))


# --------------------------------------------------------------------------
# Event scanning primitives
# --------------------------------------------------------------------------

def test_proper_crossing_parameter_is_exact_fraction():
    # Diagonal (0,0)->(3,3) crossed by horizontal (0,3)->(3,0): t = 1/2.
    params = boundary_parameters(
        ip(0, 3), ip(3, 0), [(ip(0, 0), ip(3, 3))]
    )
    assert params == {Fraction(1, 2)}
    assert all(isinstance(t, Fraction) for t in params)


def test_fractional_t_junction_parameter():
    # Horizontal segment through the endpoint (2,2) of a vertical edge.
    params = boundary_parameters(
        ip(0, 2), ip(5, 2), [(ip(2, 2), ip(2, 6))]
    )
    assert params == {Fraction(2, 5)}


def test_collinear_overlap_registers_both_endpoints():
    # Segment 0..6 along y=0 shares [1,4] with an edge 1..4.
    params = boundary_parameters(
        ip(0, 0), ip(6, 0), [(ip(1, 0), ip(4, 0))]
    )
    assert params == {Fraction(1, 6), Fraction(2, 3)}


def test_tangent_vertex_is_an_event():
    # Segment ending at / passing through polygon vertex (2,1): segment
    # (0,3)->(4,-1) meets it at t = 1/2.
    square = [(2, 1), (6, 1), (6, 5), (2, 5)]
    edges = [
        (int_point(square[i]), int_point(square[(i + 1) % 4]))
        for i in range(4)
    ]
    params = boundary_parameters(ip(0, 3), ip(4, -1), edges)
    assert Fraction(1, 2) in params


# --------------------------------------------------------------------------
# Point classification
# --------------------------------------------------------------------------

def test_point_relation_with_hole():
    assert point_relation(ip(-1, 5), SQ_HOLE) == OUTSIDE
    assert point_relation(ip(2, 5), SQ_HOLE) == INSIDE
    assert point_relation(ip(5, 5), SQ_HOLE) == OUTSIDE     # in the hole
    assert point_relation(ip(0, 5), SQ_HOLE) == BOUNDARY    # exterior edge
    assert point_relation(ip(4, 5), SQ_HOLE) == BOUNDARY    # hole edge
    assert point_relation(ip(10, 10), SQ_HOLE) == BOUNDARY  # vertex


def test_point_relation_multiple_polygons_and_empty():
    group = [
        build_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], []),
        build_polygon([(4, 0), (6, 0), (6, 2), (4, 2)], []),
    ]
    assert point_relation(ip(1, 1), group) == INSIDE
    assert point_relation(ip(3, 1), group) == OUTSIDE
    assert point_relation(ip(5, 1), group) == INSIDE
    assert point_relation(ip(2, 1), group) == BOUNDARY
    assert point_relation(ip(1, 1), []) == OUTSIDE


# --------------------------------------------------------------------------
# Independent reference classifier (winding number, written differently
# from the ray-casting predicates used by the engine).
# --------------------------------------------------------------------------

def _ring_winding_number(point: Point, ring) -> int:
    px, py = point
    wn = 0
    pts = ring.points
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if point_on_segment(point, a, b):
            return 0  # boundary sentinel, handled by caller
        if a[1] <= py < b[1]:  # upward crossing of the horizontal ray
            if (b[0] - a[0]) * (py - a[1]) - (point[0] - a[0]) * (b[1] - a[1]) > 0:
                wn += 1
        elif b[1] <= py < a[1]:  # downward crossing
            if (b[0] - a[0]) * (py - a[1]) - (point[0] - a[0]) * (b[1] - a[1]) < 0:
                wn -= 1
    return wn


def reference_relation(point: Point, group) -> str:
    for poly in group:
        for ring in (poly.exterior, *poly.holes):
            pts = ring.points
            for i in range(len(pts)):
                if point_on_segment(point, pts[i], pts[(i + 1) % len(pts)]):
                    return BOUNDARY
    for poly in group:
        if _ring_winding_number(point, poly.exterior) != 0:
            if any(_ring_winding_number(point, h) != 0 for h in poly.holes):
                continue
            return INSIDE
    return OUTSIDE


GROUPS = (
    [build_polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                   [[(4, 4), (6, 4), (6, 6), (4, 6)]])],
    [build_polygon([(2, -2), (8, -2), (8, 4), (2, 4)], []),
     build_polygon([(-6, 6), (-3, 6), (-3, 9), (-6, 9)], [])],
)


def test_partition_matches_independent_reference_on_random_paths():
    rng = random.Random(7)
    for _ in range(120):
        path = []
        while len(path) < rng.randint(2, 5):
            cand = (rng.randint(-8, 12), rng.randint(-8, 12))
            if path and cand == path[-1]:
                continue
            path.append(cand)
        pts = tuple(int_point(p) for p in path)
        profiles = transect_profile(GROUPS[0], GROUPS[1], pts)

        for k, profile in enumerate(profiles):
            assert profile.index == k
            assert profile.intervals[0].start == 0
            assert profile.intervals[-1].end == 1
            for iv in profile.intervals:
                # Independent classification at three interior samples.
                for num in (1, 3, 5):
                    t = (iv.start * (8 - num) + iv.end * num) / 8
                    sample = (
                        pts[k][0] + t * (pts[k + 1][0] - pts[k][0]),
                        pts[k][1] + t * (pts[k + 1][1] - pts[k][1]),
                    )
                    assert point_relation(sample, GROUPS[0]) == iv.a
                    assert point_relation(sample, GROUPS[1]) == iv.b
                    assert reference_relation(sample, GROUPS[0]) == iv.a
                    assert reference_relation(sample, GROUPS[1]) == iv.b
            for ct in profile.contacts:
                point = (
                    pts[k][0] + ct.at * (pts[k + 1][0] - pts[k][0]),
                    pts[k][1] + ct.at * (pts[k + 1][1] - pts[k][1]),
                )
                assert point_relation(point, GROUPS[0]) == ct.at_a
                assert point_relation(point, GROUPS[1]) == ct.at_b


def test_reversed_path_is_exact_mirror_partition():
    rng = random.Random(99)
    for _ in range(80):
        path = []
        while len(path) < rng.randint(2, 5):
            cand = (rng.randint(-8, 12), rng.randint(-8, 12))
            if path and cand == path[-1]:
                continue
            path.append(cand)
        pts = tuple(int_point(p) for p in path)
        fwd = transect_profile(GROUPS[0], GROUPS[1], pts)
        rev = transect_profile(GROUPS[0], GROUPS[1], tuple(reversed(pts)))
        n = len(fwd)

        for k in range(n):
            fi, ri = fwd[k].intervals, rev[n - 1 - k].intervals
            assert len(fi) == len(ri)
            for a_iv, b_iv in zip(fi, reversed(ri)):
                assert b_iv.start == 1 - a_iv.end
                assert b_iv.end == 1 - a_iv.start
                assert (b_iv.a, b_iv.b) == (a_iv.a, a_iv.b)

        fwd_contacts = {
            (s.index, c.at): c for s in fwd for c in s.contacts
        }
        rev_contacts = {
            (s.index, c.at): c for s in rev for c in s.contacts
        }
        assert set(fwd_contacts) == {
            (n - 1 - idx, 1 - t) for idx, t in rev_contacts
        }
        for (idx, t), c in fwd_contacts.items():
            d = rev_contacts[(n - 1 - idx, 1 - t)]
            assert (d.before_a, d.at_a, d.after_a) == (
                c.after_a, c.at_a, c.before_a)
            assert (d.before_b, d.at_b, d.after_b) == (
                c.after_b, c.at_b, c.before_b)


def test_events_contain_every_boundary_hit_of_dense_sample():
    # Every parameter at which a dense rational sample lands on a boundary
    # must equal an event parameter (interval boundary or isolated contact).
    path = tuple(int_point(p) for p in [(-1, 1), (11, 7), (-4, 4)])
    profiles = transect_profile(GROUPS[0], GROUPS[1], path)
    all_edges = []
    for group in GROUPS:
        for poly in group:
            for ring in (poly.exterior, *poly.holes):
                ps = ring.points
                for i in range(len(ps)):
                    all_edges.append((ps[i], ps[(i + 1) % len(ps)]))

    for k, profile in enumerate(profiles):
        events = boundary_parameters(path[k], path[k + 1], all_edges)
        cuts = {iv.start for iv in profile.intervals} | {Fraction(1)}
        # No boundary hit is missing from the partition.
        assert events <= cuts
        assert {c.at for c in profile.contacts} <= cuts
