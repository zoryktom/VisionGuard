"""Geometry helpers."""

from visionguard.utils.geometry import iou_xyxy, point_in_polygon


def test_iou_identical_is_one() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou_xyxy(box, box) == 1.0


def test_iou_disjoint_is_zero() -> None:
    assert iou_xyxy((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_point_in_polygon() -> None:
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((20, 20), square)
