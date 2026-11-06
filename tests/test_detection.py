"""Tests for TeamClassifier, BallTracker, TacticalBoard, and device resolution."""

import numpy as np
import pytest

from futsal_analytics.board import TacticalBoard
from futsal_analytics.config import Config
from futsal_analytics.detection import BallTracker, TeamClassifier, resolve_device


class _Dets:
    """Minimal stand-in for supervision.Detections: just `xyxy` and `len`."""

    def __init__(self, xyxy):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)

    def __len__(self):
        return len(self.xyxy)


def _kit_frame(kits_bgr):
    """One 40x100 'player' per kit colour, side by side on a green pitch."""
    n = len(kits_bgr)
    frame = np.zeros((120, 50 * n + 10, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)
    boxes = []
    for i, bgr in enumerate(kits_bgr):
        x1 = 5 + 50 * i
        frame[10:110, x1 : x1 + 40] = bgr
        boxes.append([x1, 10, x1 + 40, 110])
    return frame, _Dets(boxes)


# ---------------------------------------------------------------------------
# BallTracker
# ---------------------------------------------------------------------------


class TestBallTracker:
    def test_none_input_returns_none(self):
        bt = BallTracker()
        assert bt.update(None) is None

    def test_single_position_returned(self):
        bt = BallTracker()
        pos = np.array([100.0, 200.0])
        result = bt.update(pos)
        np.testing.assert_allclose(result, pos)

    def test_smoothing_averages_positions(self):
        # Use a large max_distance so history is not reset between steps
        bt = BallTracker(max_distance=1000)
        bt.update(np.array([0.0, 0.0]))
        bt.update(np.array([100.0, 100.0]))
        result = bt.update(np.array([200.0, 200.0]))
        np.testing.assert_allclose(result, [100.0, 100.0])

    def test_large_jump_resets_history(self):
        bt = BallTracker(max_distance=50)
        bt.update(np.array([0.0, 0.0]))
        result = bt.update(np.array([1000.0, 1000.0]))
        np.testing.assert_allclose(result, [1000.0, 1000.0])

    def test_none_clears_history(self):
        """After a missed frame, smoothing should start fresh, not avg stale positions."""
        bt = BallTracker()
        bt.update(np.array([0.0, 0.0]))
        bt.update(np.array([10.0, 10.0]))
        bt.update(None)
        result = bt.update(np.array([500.0, 500.0]))
        # History was cleared by the None — result should equal the new position alone
        np.testing.assert_allclose(result, [500.0, 500.0])

    def test_ball_seen_count_increments(self):
        bt = BallTracker()
        bt.update(np.array([1.0, 1.0]))
        bt.update(np.array([2.0, 2.0]))
        assert bt.ball_seen_count == 2

    def test_none_does_not_increment_count(self):
        bt = BallTracker()
        bt.update(None)
        assert bt.ball_seen_count == 0


# ---------------------------------------------------------------------------
# TeamClassifier
# ---------------------------------------------------------------------------


class TestTeamClassifierColorFeatures:
    def test_returns_four_element_vector(self, synthetic_frame):
        # [sin_H, cos_H, S, V]: hue is circular, so it takes two components.
        tc = TeamClassifier()
        bbox = np.array([10, 10, 60, 80], dtype=np.float32)
        features = tc.get_jersey_color_features(synthetic_frame, bbox)
        assert features.shape == (4,)
        assert features.dtype == np.float32

    def test_saturation_and_value_are_scaled_to_unit_range(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:] = (0, 0, 255)  # pure red in BGR
        features = TeamClassifier().get_jersey_color_features(
            frame, np.array([0, 0, 90, 90], dtype=np.float32)
        )
        assert np.all(np.abs(features) <= 1.0 + 1e-6)
        assert features[2] == pytest.approx(1.0) and features[3] == pytest.approx(1.0)

    def test_red_hue_wraparound_is_not_green(self):
        """Red sits at H~0 and H~180; (sin, cos) must stay near angle 0, not drift to 90 (green)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :50] = (0, 0, 255)  # H = 0
        frame[:, 50:] = (20, 0, 255)  # H just under 180
        features = TeamClassifier().get_jersey_color_features(
            frame, np.array([0, 0, 100, 100], dtype=np.float32)
        )
        assert features[1] > 0.9  # cos(H) ~ 1: red, not the cos ~ -1 a plain mean of H would give

    def test_empty_crop_is_treated_as_black(self, synthetic_frame):
        # _safe_crop never returns a zero-size array: a degenerate box becomes a 1x1 black pixel,
        # i.e. hue angle 0 -> (sin, cos) = (0, 1), and S = V = 0.
        tc = TeamClassifier()
        bbox = np.array([10, 10, 10, 10], dtype=np.float32)
        features = tc.get_jersey_color_features(synthetic_frame, bbox)
        np.testing.assert_array_equal(features, [0, 1, 0, 0])

    def test_train_teams_fits_distinct_kits_and_separates_them(self):
        frame, dets = _kit_frame([(0, 0, 255)] * 5 + [(255, 0, 0)] * 5 + [(0, 255, 255)] * 2)
        tc = TeamClassifier()
        assert tc.train_teams(frame, dets, Config()) is True
        red = {tc.predict_team(frame, b) for b in dets.xyxy[:5]}
        blue = {tc.predict_team(frame, b) for b in dets.xyxy[5:10]}
        assert len(red) == 1 and len(blue) == 1 and red != blue
        assert -1 not in red | blue  # the two big groups are teams; the small one is the referee

    def test_train_teams_refuses_identical_kits(self):
        frame, dets = _kit_frame([(0, 0, 255)] * 8)
        tc = TeamClassifier()
        assert tc.train_teams(frame, dets, Config()) is False
        assert tc.trained is False

    def test_train_teams_works_for_white_vs_black_kits(self):
        frame, dets = _kit_frame([(255, 255, 255)] * 5 + [(10, 10, 10)] * 5 + [(0, 0, 255)] * 2)
        assert TeamClassifier().train_teams(frame, dets, Config()) is True

    def test_train_teams_needs_enough_players(self):
        frame, dets = _kit_frame([(0, 0, 255)] * 2 + [(255, 0, 0)] * 2)
        assert TeamClassifier().train_teams(frame, dets, Config()) is False

    def test_predict_team_before_training_returns_zero(self, synthetic_frame):
        tc = TeamClassifier()
        bbox = np.array([10, 10, 60, 80], dtype=np.float32)
        assert tc.predict_team(synthetic_frame, bbox) == 0

    def test_safe_crop_clamps_out_of_bounds(self, synthetic_frame):
        tc = TeamClassifier()
        crop = tc._safe_crop(synthetic_frame, -100, -100, 9999, 9999)
        h, w = synthetic_frame.shape[:2]
        assert crop.shape[0] <= h
        assert crop.shape[1] <= w


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        # Returns "cuda" verbatim — caller is responsible for availability
        assert resolve_device("cuda") == "cuda"

    def test_auto_returns_valid_device(self):
        result = resolve_device("auto")
        assert result in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# TacticalBoard
# ---------------------------------------------------------------------------


class TestTacticalBoard:
    def test_draw_state_returns_correct_shape(self):
        board = TacticalBoard(700, 350)
        img = board.draw_state([], None)
        assert img.shape == (350, 700, 3)

    def test_draw_state_with_players(self):
        board = TacticalBoard(700, 350)
        players = [
            (1, np.array([350.0, 175.0]), 0),
            (2, np.array([100.0, 100.0]), 1),
        ]
        img = board.draw_state(players, np.array([200.0, 150.0]))
        assert img.shape == (350, 700, 3)

    def test_referee_uses_grey_color(self):
        board = TacticalBoard(700, 350)
        # Referee should be drawn (unlike before), in grey
        img_no_players = board.draw_state([], None)
        img_referee = board.draw_state([(99, np.array([350.0, 175.0]), -1)], None)
        # The referee is now drawn, so the images differ
        assert not np.array_equal(img_no_players, img_referee)

    def test_team_colors_defined(self):
        assert len(TacticalBoard.TEAM_COLORS) >= 2

    def test_metres_per_pixel_scale(self):
        board = TacticalBoard(700, 350)
        # Pitch is 40m × 20m
        assert board.m_per_px_x == pytest.approx(40 / 700)
        assert board.m_per_px_y == pytest.approx(20 / 350)
