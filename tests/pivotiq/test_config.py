"""PivotIQConfig loading and validation: bad input must fail loudly and say why."""

import pytest

from pivotiq.config import ConfigError, PivotIQConfig


def write(tmp_path, text):
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    return p


class TestDefaults:
    def test_default_config_is_valid(self):
        PivotIQConfig.default().validate()

    def test_empty_file_keeps_defaults(self, tmp_path):
        assert PivotIQConfig.from_toml(write(tmp_path, "")) == PivotIQConfig.default()

    def test_partial_override_keeps_other_defaults(self, tmp_path):
        cfg = PivotIQConfig.from_toml(write(tmp_path, "[detection]\nplayer_confidence = 0.55\n"))
        assert cfg.detection.player_confidence == 0.55
        assert cfg.detection.ball_confidence == PivotIQConfig.default().detection.ball_confidence
        assert cfg.events == PivotIQConfig.default().events

    def test_int_is_accepted_for_float_field(self, tmp_path):
        cfg = PivotIQConfig.from_toml(write(tmp_path, "[events]\nshot_speed_threshold_mps = 7\n"))
        assert cfg.events.shot_speed_threshold_mps == 7

    def test_optional_ball_model_accepts_a_path_string(self, tmp_path):
        cfg = PivotIQConfig.from_toml(write(tmp_path, '[detection]\nball_model = "ball.pt"\n'))
        assert cfg.detection.ball_model == "ball.pt"


class TestFileProblems:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            PivotIQConfig.from_toml(tmp_path / "nope.toml")

    def test_invalid_toml(self, tmp_path):
        with pytest.raises(ConfigError, match="invalid TOML"):
            PivotIQConfig.from_toml(write(tmp_path, "[detection\nx ="))

    def test_unknown_section(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown section"):
            PivotIQConfig.from_toml(write(tmp_path, "[detecton]\nplayer_confidence = 0.5\n"))

    def test_unknown_key_names_the_key_and_the_valid_ones(self, tmp_path):
        with pytest.raises(ConfigError, match="player_confidnce") as exc:
            PivotIQConfig.from_toml(write(tmp_path, "[detection]\nplayer_confidnce = 0.5\n"))
        assert "player_confidence" in str(exc.value)

    def test_section_must_be_a_table(self, tmp_path):
        with pytest.raises(ConfigError, match="must be a table"):
            PivotIQConfig.from_toml(write(tmp_path, "detection = 3\n"))


class TestValues:
    @pytest.mark.parametrize(
        "toml",
        [
            "[detection]\nplayer_confidence = 1.5\n",
            "[detection]\nplayer_confidence = -0.1\n",
            "[detection]\nball_confidence = 2\n",
            '[detection]\ndevice = "gpu"\n',
            "[tracking]\ntrack_thresh = 3.0\n",
            "[tracking]\ntrack_buffer = 0\n",
            "[tracking]\nmin_box_area = -1.0\n",
            "[registration]\nmin_visible_keypoints = 3\n",
            "[registration]\nransac_reproj_threshold = 0.0\n",
            "[registration]\ntemporal_smoothing_alpha = 1.2\n",
            "[registration]\noptical_flow_max_corners = 0\n",
            "[events]\nshot_speed_threshold_mps = 0\n",
            "[events]\nchance_xg_threshold = 1.5\n",
            "[events]\nbuildup_seconds = -5\n",
            "[events]\ngoal_rebound_frames = -1\n",
        ],
    )
    def test_out_of_range_values_are_rejected(self, tmp_path, toml):
        with pytest.raises(ConfigError):
            PivotIQConfig.from_toml(write(tmp_path, toml))

    @pytest.mark.parametrize(
        "toml",
        [
            '[detection]\nplayer_confidence = "high"\n',
            "[detection]\nplayer_confidence = true\n",
            "[detection]\nplayer_model = 3\n",
            "[tracking]\ntrack_buffer = 2.5\n",
            "[registration]\nmin_visible_keypoints = true\n",
            "[events]\nbuildup_seconds = nan\n",
            "[events]\nbuildup_seconds = inf\n",
        ],
    )
    def test_wrong_types_are_rejected(self, tmp_path, toml):
        with pytest.raises(ConfigError):
            PivotIQConfig.from_toml(write(tmp_path, toml))

    def test_boundaries_are_inclusive(self, tmp_path):
        cfg = PivotIQConfig.from_toml(
            write(tmp_path, "[detection]\nplayer_confidence = 0.0\nball_confidence = 1.0\n"
                            "[registration]\nmin_visible_keypoints = 4\n")
        )
        assert cfg.registration.min_visible_keypoints == 4
