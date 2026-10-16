"""Edge cases for homography maths and pitch geometry: degenerate H, points off the pitch."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from pivotiq.geometry import (
    LEFT_GOAL_POLYGON,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    point_in_polygon,
)
from pivotiq.geometry.pitch import all_keypoints_array
from pivotiq.models import Homography

PITCH_POLYGON = np.array(
    [(0, 0), (PITCH_LENGTH_M, 0), (PITCH_LENGTH_M, PITCH_WIDTH_M), (0, PITCH_WIDTH_M)],
    dtype=np.float32,
)


def real_H() -> np.ndarray:
    """A plausible camera -> pitch homography fitted from 4 known correspondences."""
    import cv2

    src = np.array([[100, 50], [1180, 60], [1250, 680], [30, 670]], dtype=np.float32)
    dst = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


class TestDegenerateHomography:
    @pytest.mark.parametrize(
        "H",
        [
            np.zeros((3, 3)),
            np.ones((3, 3)),                              # rank 1
            np.array([[1, 2, 3], [2, 4, 6], [0, 0, 1]]),   # dependent rows
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),   # collapses the plane
        ],
    )
    def test_singular_matrices_are_rejected(self, H):
        with pytest.raises(ValueError, match="singular"):
            Homography.from_numpy(0, H.astype(float))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_entries_are_rejected(self, bad):
        H = np.eye(3)
        H[0, 2] = bad
        with pytest.raises(ValueError, match="finite"):
            Homography.from_numpy(0, H)

    def test_json_with_singular_matrix_is_rejected_on_load(self):
        with pytest.raises(ValidationError):
            Homography.model_validate_json('{"frame_idx": 0, "H": [[0,0,0],[0,0,0],[0,0,0]]}')

    def test_point_on_the_vanishing_line_raises_instead_of_returning_inf(self):
        # Third row (1, 0, -5): w = x - 5, so every point with x = 5 maps to infinity.
        h = Homography.from_numpy(0, np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, -5.0]]))
        with pytest.raises(ValueError, match="infinity"):
            h.pixel_to_pitch(5.0, 3.0)
        assert h.pixel_to_pitch(6.0, 3.0) == pytest.approx((6.0, 3.0))

    def test_nearly_singular_but_valid_matrix_still_works(self):
        h = Homography.from_numpy(0, np.diag([1.0, 1.0, 1e-3]))
        x, y = h.pixel_to_pitch(2.0, 4.0)
        assert math.isclose(x, 2000.0) and math.isclose(y, 4000.0)


class TestRealHomography:
    def test_roundtrip_pixel_pitch_pixel(self):
        h = Homography.from_numpy(0, real_H())
        for px, py in [(100, 50), (640, 360), (1250, 680), (-50, 900)]:
            back = h.pitch_to_pixel(*h.pixel_to_pitch(px, py))
            assert back == pytest.approx((px, py), abs=1e-6)

    def test_pitch_corners_map_back_to_the_image_corners(self):
        h = Homography.from_numpy(0, real_H())
        assert h.pitch_to_pixel(0, 0) == pytest.approx((100, 50), abs=1e-3)
        assert h.pitch_to_pixel(40, 20) == pytest.approx((1250, 680), abs=1e-3)

    def test_image_corner_maps_to_pitch_corner(self):
        h = Homography.from_numpy(0, real_H())
        assert h.pixel_to_pitch(1180, 60) == pytest.approx((40, 0), abs=1e-3)

    def test_a_point_outside_the_image_lands_outside_the_pitch(self):
        h = Homography.from_numpy(0, real_H())
        x, y = h.pixel_to_pitch(-500, 700)
        assert not point_in_polygon((x, y), PITCH_POLYGON)

    def test_centre_of_the_image_is_inside_the_pitch(self):
        h = Homography.from_numpy(0, real_H())
        assert point_in_polygon(h.pixel_to_pitch(640, 370), PITCH_POLYGON)


class TestPointsOffThePitch:
    @pytest.mark.parametrize(
        "pt",
        [(-0.01, 10), (40.01, 10), (20, -0.01), (20, 20.01), (-100, -100), (1e9, 1e9)],
    )
    def test_points_outside_the_pitch_rectangle(self, pt):
        assert not point_in_polygon(pt, PITCH_POLYGON)

    @pytest.mark.parametrize("pt", [(20, 10), (0.01, 0.01), (39.99, 19.99)])
    def test_points_inside_the_pitch_rectangle(self, pt):
        assert point_in_polygon(pt, PITCH_POLYGON)

    def test_every_canonical_keypoint_is_on_or_inside_the_pitch(self):
        for x, y in all_keypoints_array():
            assert -1e-6 <= x <= PITCH_LENGTH_M + 1e-6 and -1e-6 <= y <= PITCH_WIDTH_M + 1e-6

    def test_nan_point_is_never_inside(self):
        assert not point_in_polygon((math.nan, 10.0), PITCH_POLYGON)
        assert not point_in_polygon((10.0, math.nan), PITCH_POLYGON)

    def test_degenerate_polygons_do_not_crash(self):
        assert not point_in_polygon((0, 0), np.empty((0, 2), dtype=np.float32))
        assert not point_in_polygon((0, 0), np.array([(0, 0)], dtype=np.float32))
        flat = np.array([(0, 0), (1, 0), (2, 0)], dtype=np.float32)  # zero area
        assert not point_in_polygon((1, 0.5), flat)

    def test_ball_behind_the_goal_line_is_in_the_goal_net_polygon(self):
        assert point_in_polygon((-0.5, 10.0), LEFT_GOAL_POLYGON)
        assert not point_in_polygon((-0.5, 12.0), LEFT_GOAL_POLYGON)  # wide of the post
