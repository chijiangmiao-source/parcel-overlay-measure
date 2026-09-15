"""Exact traversal ("transect") profiles of polyline segments.

Given the two validated multi-polygon groups and a polyline of exact rational
vertices, every original segment ``P -> Q`` is split at *every* parameter
``t`` (``0 <= t <= 1``) at which the segment meets a region boundary of either
group: proper crossings, endpoint / vertex contacts, T-junctions and the end
points of collinear boundary overlaps.  Between two consecutive event
parameters the open sub-segment has a constant relation (outside / inside /
boundary) to each group, so the resulting intervals form an ordered partition
of ``[0, 1]`` with no gap and no overlap.

An isolated boundary contact (a single event point whose neighbouring open
sub-segments are not themselves boundary runs) additionally records the
before / at / after relation for each group; the missing side of a contact at
an endpoint of the whole polyline is ``None``.

All parameters and event points are :class:`fractions.Fraction`; intersection
parameters are never converted to floating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Set, Tuple

from .arrangement import active_rings
from .rationals import Point, cross, point_on_segment, point_in_interior
from .validation import Polygon, Ring

OUTSIDE = "outside"
INSIDE = "inside"
BOUNDARY = "boundary"

Relation = str  # one of OUTSIDE / INSIDE / BOUNDARY

ZERO = Fraction(0)
ONE = Fraction(1)


@dataclass(frozen=True)
class Interval:
    """Half-open parameter range ``[start, end]`` with constant group states."""

    start: Fraction
    end: Fraction
    a: Relation
    b: Relation


@dataclass(frozen=True)
class Contact:
    """An isolated boundary contact at parameter ``at``.

    The group triples give the relation immediately before, exactly at and
    immediately after the contact point; the side missing at a polyline
    endpoint is ``None``.
    """

    at: Fraction
    before_a: Optional[Relation]
    at_a: Relation
    after_a: Optional[Relation]
    before_b: Optional[Relation]
    at_b: Relation
    after_b: Optional[Relation]


@dataclass(frozen=True)
class SegmentProfile:
    index: int
    intervals: Tuple[Interval, ...]
    contacts: Tuple[Contact, ...]


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------

def _active_parts(poly: Polygon) -> Tuple[Ring, Tuple[Ring, ...]]:
    """Exterior ring and the hole rings that actually bound material area.

    Mirrors the area engine: a hole nested strictly inside another hole is not
    a boundary of the material region (set-difference semantics).
    """
    rings = [ring for ring, _semantic in active_rings(poly)]
    return rings[0], tuple(rings[1:])


def _group_boundary_edges(
    group: Sequence[Polygon],
) -> Tuple[Tuple[Point, Point], ...]:
    edges: List[Tuple[Point, Point]] = []
    for poly in group:
        exterior, holes = _active_parts(poly)
        for ring in (exterior, *holes):
            pts = ring.points
            n = len(pts)
            for i in range(n):
                edges.append((pts[i], pts[(i + 1) % n]))
    return tuple(edges)


def point_relation(point: Point, group: Sequence[Polygon]) -> Relation:
    """Outside / inside / boundary of one group's material region."""
    # Boundary first: polygons of one group never touch, so a point on any
    # active ring is a boundary point of the union regardless of interiors.
    for poly in group:
        exterior, holes = _active_parts(poly)
        for ring in (exterior, *holes):
            pts = ring.points
            n = len(pts)
            for i in range(n):
                if point_on_segment(point, pts[i], pts[(i + 1) % n]):
                    return BOUNDARY
    for poly in group:
        exterior, holes = _active_parts(poly)
        if point_in_interior(point, exterior.points):
            if any(point_in_interior(point, h.points) for h in holes):
                continue  # inside a hole: outside the material region
            return INSIDE
    return OUTSIDE


# ---------------------------------------------------------------------------
# Event scanning: every parameter at which segment P->Q meets a boundary
# ---------------------------------------------------------------------------

def boundary_parameters(
    p: Point, q: Point, edges: Sequence[Tuple[Point, Point]]
) -> Set[Fraction]:
    """All parameters ``t`` with ``P + t*(Q-P)`` on one of the closed edges."""
    params: Set[Fraction] = set()
    rx, ry = q[0] - p[0], q[1] - p[1]
    r2 = rx * rx + ry * ry  # > 0: consecutive polyline points are distinct
    for a, b in edges:
        sx, sy = b[0] - a[0], b[1] - a[1]
        denom = rx * sy - ry * sx
        if denom != 0:
            # Non-parallel supporting lines: t = ((a - P) x s) / (r x s)
            cax, cay = a[0] - p[0], a[1] - p[1]
            t = (cax * sy - cay * sx) / denom
            if t < ZERO or t > ONE:
                continue
            s = (cax * ry - cay * rx) / denom
            if ZERO <= s <= ONE:
                params.add(t)
            continue
        # Parallel: only a collinear edge can share points with the segment.
        if cross(p, q, a) != 0 or cross(p, q, b) != 0:
            continue
        ta = ((a[0] - p[0]) * rx + (a[1] - p[1]) * ry) / r2
        tb = ((b[0] - p[0]) * rx + (b[1] - p[1]) * ry) / r2
        lo, hi = (ta, tb) if ta <= tb else (tb, ta)
        lo = max(ZERO, lo)
        hi = min(ONE, hi)
        if lo <= hi:
            # Both ends of the (possibly zero-length) boundary overlap are
            # events; a proper collinear run becomes one BOUNDARY interval.
            params.add(lo)
            params.add(hi)
    return params


def _point_at(p: Point, q: Point, t: Fraction) -> Point:
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _is_isolated(
    side_left: Optional[Relation], side_right: Optional[Relation], at: Relation
) -> bool:
    """An isolated contact: boundary only at the single event point."""
    if at != BOUNDARY:
        return False
    if side_left is not None and side_left == BOUNDARY:
        return False
    if side_right is not None and side_right == BOUNDARY:
        return False
    return True


def transect_profile(
    group_a: Sequence[Polygon],
    group_b: Sequence[Polygon],
    path: Sequence[Point],
) -> Tuple[SegmentProfile, ...]:
    if len(path) < 2:
        raise ValueError("transect path needs at least two points")
    for i in range(1, len(path)):
        if path[i] == path[i - 1]:
            raise ValueError(f"path point {i} repeats point {i - 1}")

    edges = _group_boundary_edges(group_a) + _group_boundary_edges(group_b)

    n_segments = len(path) - 1
    interval_table: List[List[Interval]] = []
    events_table: List[List[Fraction]] = []

    for k in range(n_segments):
        p, q = path[k], path[k + 1]
        params = {ZERO, ONE}
        params |= boundary_parameters(p, q, edges)
        ts = sorted(params)
        events_table.append(ts)

        intervals: List[Interval] = []
        for j in range(len(ts) - 1):
            lo, hi = ts[j], ts[j + 1]
            # The midpoint of two consecutive events never lies on a boundary
            # unless the whole open range is a collinear boundary run; both are
            # classified exactly.
            mid = (lo + hi) / 2
            mp = _point_at(p, q, mid)
            intervals.append(
                Interval(
                    lo,
                    hi,
                    point_relation(mp, group_a),
                    point_relation(mp, group_b),
                )
            )
        interval_table.append(intervals)

    profiles: List[SegmentProfile] = []
    for k in range(n_segments):
        p, q = path[k], path[k + 1]
        ts = events_table[k]
        intervals = interval_table[k]
        contacts: List[Contact] = []

        for j, t in enumerate(ts):
            point = _point_at(p, q, t)
            at_a = point_relation(point, group_a)
            at_b = point_relation(point, group_b)

            # Relation on the open side immediately before / after the event,
            # continuing into the adjacent original segment at its ends.
            if j > 0:
                before_a, before_b = intervals[j - 1].a, intervals[j - 1].b
            elif k > 0:
                before_a = interval_table[k - 1][-1].a
                before_b = interval_table[k - 1][-1].b
            else:
                before_a = before_b = None

            if j < len(ts) - 1:
                after_a, after_b = intervals[j].a, intervals[j].b
            elif k + 1 < n_segments:
                after_a = interval_table[k + 1][0].a
                after_b = interval_table[k + 1][0].b
            else:
                after_a = after_b = None

            if (_is_isolated(before_a, after_a, at_a)
                    or _is_isolated(before_b, after_b, at_b)):
                contacts.append(
                    Contact(t, before_a, at_a, after_a,
                            before_b, at_b, after_b)
                )

        profiles.append(SegmentProfile(k, tuple(intervals), tuple(contacts)))
    return tuple(profiles)
