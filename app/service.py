"""Service layer: request geometry -> validated polygons -> exact results."""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence

from .geometry.rationals import int_point
from .geometry.transect import (
    Contact,
    Interval,
    SegmentProfile,
    transect_profile,
)
from .geometry.arrangement import overlap_area
from .geometry.validation import (
    GeometryValidationError,
    Polygon,
    build_polygon,
    validate_group,
)
from .models import PolygonIn, TransectRequest


def build_group(polys: Sequence[PolygonIn], name: str) -> List[Polygon]:
    built = []
    for i, p in enumerate(polys):
        try:
            built.append(
                build_polygon(p.exterior, [list(h) for h in p.holes])
            )
        except GeometryValidationError as exc:
            exc.loc = (name, i, *exc.loc)
            raise
    validate_group(built, name)
    return built


def compute_overlap(a: Sequence[PolygonIn], b: Sequence[PolygonIn]) -> Fraction:
    group_a = build_group(a, "a")
    # Validate group B independently as well; cross-group contact is legal
    # and always contributes zero area.
    group_b = build_group(b, "b")
    area = overlap_area(group_a, group_b)
    if area < 0:  # pragma: no cover - defensive; overlap is non-negative
        area = -area
    return area


def round_half_up_thirds(value: Fraction) -> str:
    """Format a non-negative fraction rounded half-up to 3 decimal places."""
    if value < 0:  # pragma: no cover - defensive
        raise ValueError("only non-negative areas are supported")
    scale = 1000
    scaled, rem = divmod(value.numerator * scale, value.denominator)
    if rem * 2 >= value.denominator:
        scaled += 1
    whole, frac = divmod(scaled, scale)
    return f"{whole}.{frac:03d}"


def _fraction_pair(value: Fraction) -> List[int]:
    """Reduced ``[numerator, denominator]``; Fraction is already reduced."""
    return [value.numerator, value.denominator]


def _interval_payload(interval: Interval) -> Dict[str, object]:
    return {
        "start": _fraction_pair(interval.start),
        "end": _fraction_pair(interval.end),
        "a": interval.a,
        "b": interval.b,
    }


def _contact_payload(contact: Contact) -> Dict[str, object]:
    return {
        "at": _fraction_pair(contact.at),
        "before_a": contact.before_a,
        "at_a": contact.at_a,
        "after_a": contact.after_a,
        "before_b": contact.before_b,
        "at_b": contact.at_b,
        "after_b": contact.after_b,
    }


def _segment_payload(profile: SegmentProfile) -> Dict[str, object]:
    return {
        "segment": profile.index,
        "intervals": [_interval_payload(i) for i in profile.intervals],
        "contacts": [_contact_payload(c) for c in profile.contacts],
    }


def compute_transect(req: TransectRequest) -> Dict[str, object]:
    """Validate both groups exactly like the overlap API and profile the path.

    The geometry layer partitions every original segment into intervals whose
    parameters cover ``[0, 1]`` exactly; this is re-checked here as a service
    invariant before serialization.
    """
    group_a = build_group(req.a, "a")
    group_b = build_group(req.b, "b")
    path = tuple(int_point((int(x), int(y))) for x, y in req.path)

    profiles = transect_profile(group_a, group_b, path)

    segments: List[Dict[str, object]] = []
    for profile in profiles:
        intervals = profile.intervals
        if not intervals or intervals[0].start != 0 or intervals[-1].end != 1:
            raise RuntimeError(  # pragma: no cover - defensive
                "transect intervals do not cover parameters 0 through 1"
            )
        for prev, nxt in zip(intervals, intervals[1:]):
            if prev.end != nxt.start:
                raise RuntimeError(  # pragma: no cover - defensive
                    "transect intervals are not contiguous"
                )
        segments.append(_segment_payload(profile))
    return {"segments": segments}


__all__ = [
    "GeometryValidationError",
    "compute_overlap",
    "compute_transect",
    "round_half_up_thirds",
]
