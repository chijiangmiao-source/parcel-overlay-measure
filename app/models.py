"""Pydantic request/response models for the overlap and transect APIs."""

from __future__ import annotations

from typing import Annotated, List, Literal, Tuple

from pydantic import AfterValidator, BaseModel, Field, StrictInt, field_validator

COORD_MIN = -1_000_000
COORD_MAX = 1_000_000
MIN_RING_POINTS = 3
MIN_PATH_POINTS = 2

Coordinate = Tuple[int, int]


def _check_coord_range(pair: Tuple[int, int]) -> Tuple[int, int]:
    x, y = pair
    if not (COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX):
        raise ValueError(
            f"point ({x}, {y}) is outside the closed interval "
            f"[{COORD_MIN}, {COORD_MAX}]"
        )
    return pair


# StrictInt rejects bools and fractional/non-integral floats; the after
# validator enforces the same closed coordinate interval as polygon rings.
PathCoordinate = Annotated[
    Tuple[StrictInt, StrictInt], AfterValidator(_check_coord_range)
]

Relation = Literal["outside", "inside", "boundary"]


class PolygonIn(BaseModel):
    """One polygon: a single exterior ring and zero or more holes."""

    exterior: List[Coordinate] = Field(
        description="Exterior ring: >= 3 unique vertices, no closing duplicate.",
    )
    holes: List[List[Coordinate]] = Field(
        default_factory=list,
        description="Zero or more hole rings.",
    )

    @field_validator("exterior", mode="before")
    @classmethod
    def _check_exterior(cls, v: List[Coordinate]) -> List[Coordinate]:
        _validate_ring_points(v, loc="exterior")
        return v

    @field_validator("holes", mode="before")
    @classmethod
    def _check_holes(cls, v: List[List[Coordinate]]) -> List[List[Coordinate]]:
        # This runs before type coercion: None / a number / a string would
        # otherwise raise TypeError while iterating and escape as a 500.
        if v is None:
            raise ValueError("holes must be a list of rings (use [] for none)")
        if not isinstance(v, list):
            raise ValueError("holes must be a list of rings")
        for i, hole in enumerate(v):
            if not isinstance(hole, list):
                raise ValueError(f"holes[{i}] must be a ring (list of vertices)")
            _validate_ring_points(hole, loc="holes")
        return v


class OverlapRequest(BaseModel):
    a: List[PolygonIn] = Field(description="First multi-polygon group.")
    b: List[PolygonIn] = Field(description="Second multi-polygon group.")


class TransectRequest(BaseModel):
    a: List[PolygonIn] = Field(description="First multi-polygon group.")
    b: List[PolygonIn] = Field(description="Second multi-polygon group.")
    path: List[PathCoordinate] = Field(
        description="Polyline of >= 2 integer coordinate points; consecutive "
                    "points must differ.",
        min_length=MIN_PATH_POINTS,
    )

    @field_validator("path", mode="before")
    @classmethod
    def _check_path_shape(cls, v: List[Coordinate]) -> List[Coordinate]:
        # Before type coercion, only guard the container itself: None / a
        # number / a string would raise TypeError while coercing and escape as
        # a 500.  Malformed point elements are left to the Tuple[StrictInt,
        # StrictInt] coercion, which reports their precise position.
        if not isinstance(v, list):
            raise ValueError("path must be a list of [x, y] integer pairs")
        return v


def _validate_ring_points(points: List[Coordinate], *, loc: str) -> None:
    if not isinstance(points, list):
        raise ValueError("ring must be a list of [x, y] integer pairs")
    if len(points) < MIN_RING_POINTS:
        raise ValueError(
            f"ring must contain at least {MIN_RING_POINTS} vertices, "
            f"got {len(points)}"
        )
    seen: set[Coordinate] = set()
    for i, p in enumerate(points):
        if (
            not isinstance(p, (list, tuple))
            or len(p) != 2
            or not all(isinstance(c, int) and not isinstance(c, bool) for c in p)
        ):
            raise ValueError(f"vertex {i} must be an integer pair [x, y]")
        x, y = p
        if not (COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX):
            raise ValueError(
                f"vertex {i} ({x}, {y}) is outside the closed interval "
                f"[{COORD_MIN}, {COORD_MAX}]"
            )
        if (x, y) in seen:
            if i == len(points) - 1 and tuple(points[0]) == (x, y):
                raise ValueError(
                    f"vertex {i} repeats the first vertex; do not close the ring"
                )
            raise ValueError(f"vertex {i} ({x}, {y}) is duplicated")
        seen.add((x, y))


class OverlapResponse(BaseModel):
    area_sq_mm: List[int] = Field(
        description="Overlap area as the reduced fraction [numerator, denominator] "
                    "in square millimetres."
    )
    numerator: int
    denominator: int
    decimal: str = Field(
        description="Area rounded half-up to exactly three decimal places."
    )
    rounding: Literal["half-up"] = "half-up"
    units: Literal["mm^2"] = "mm^2"


class TransectInterval(BaseModel):
    """Half-open parameter interval [start, end) of one original segment."""

    start: List[int] = Field(
        description="Start parameter as reduced fraction [numerator, denominator]."
    )
    end: List[int] = Field(
        description="End parameter as reduced fraction [numerator, denominator]."
    )
    a: Relation = Field(description="Relation of the open interval to group A.")
    b: Relation = Field(description="Relation of the open interval to group B.")


class TransectContact(BaseModel):
    """A single isolated boundary-contact point within one original segment."""

    at: List[int] = Field(
        description="Contact parameter as reduced fraction [numerator, denominator]."
    )
    before_a: Relation | None
    at_a: Relation
    after_a: Relation | None
    before_b: Relation | None
    at_b: Relation
    after_b: Relation | None


class TransectSegment(BaseModel):
    segment: int = Field(description="Zero-based index of the original segment.")
    intervals: List[TransectInterval] = Field(
        description="Ordered, non-overlapping intervals covering parameters "
                    "0 through 1 without gaps."
    )
    contacts: List[TransectContact] = Field(
        description="Isolated boundary contact points (vertex tangencies, "
                    "fold vertices, ...), ordered by parameter."
    )


class TransectResponse(BaseModel):
    segments: List[TransectSegment] = Field(
        description="One profile per original polyline segment, in request order."
    )
