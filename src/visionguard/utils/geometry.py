"""Geometry helpers."""

from __future__ import annotations

from visionguard.types import BBoxXYXY, Point, Polygon


def iou_xyxy(a: BBoxXYXY, b: BBoxXYXY) -> float:
    """Intersection-over-union of two XYXY boxes."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box: BBoxXYXY) -> Point:
    """Return the centroid of an XYXY box."""

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""

    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test."""

    x, y = point
    inside = False
    pts = list(polygon)
    if len(pts) < 3:
        return False
    j = len(pts) - 1
    for i, (xi, yi) in enumerate(pts):
        xj, yj = pts[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def clip_box(box: BBoxXYXY, width: int, height: int) -> BBoxXYXY:
    """Clamp a box to image bounds."""

    x1, y1, x2, y2 = box
    return (
        float(max(0.0, min(x1, width - 1))),
        float(max(0.0, min(y1, height - 1))),
        float(max(0.0, min(x2, width - 1))),
        float(max(0.0, min(y2, height - 1))),
    )
