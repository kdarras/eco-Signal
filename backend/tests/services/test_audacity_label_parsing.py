"""
Unit tests for Audacity label parsing helpers (no database required).
"""
import pytest

from app.services.tabular_import_service import (
    _audacity_name_keys,
    _audacity_timestamp,
    _parse_audacity_line,
)


def test_parse_region_label_row() -> None:
    item = _parse_audacity_line("0.5\t2.3\tBird call", "recording1.txt", 1)
    assert item.start == 0.5
    assert item.end == 2.3
    assert item.low is None
    assert item.high is None
    assert item.label == "Bird call"
    assert item.media_keys == ["recording1"]


def test_parse_point_label_row() -> None:
    item = _parse_audacity_line("3.1\t", "recording1.txt", 1)
    assert item.start == 3.1
    assert item.end is None
    assert item.label is None


def test_parse_point_label_row_without_trailing_tab() -> None:
    item = _parse_audacity_line("3.1", "recording1.txt", 1)
    assert item.start == 3.1
    assert item.end is None
    assert item.label is None


def test_parse_spectral_label_row() -> None:
    item = _parse_audacity_line("1.0\t2.0\t1200\t3400\tCrustacean snaps", "recording1.txt", 1)
    assert item.start == 1.0
    assert item.end == 2.0
    assert item.low == 1200.0
    assert item.high == 3400.0
    assert item.label == "Crustacean snaps"


def test_parse_comma_separated_label_row() -> None:
    item = _parse_audacity_line("0.5,2.3,Bird call", "recording1.txt", 1)
    assert item.start == 0.5
    assert item.end == 2.3
    assert item.label == "Bird call"


@pytest.mark.parametrize("value", ["01:02:03.5", "01:02:03", "05:30"])
def test_parse_hhmmss_timestamps(value: str) -> None:
    assert _audacity_timestamp(value, "f.txt", 1) > 0


def test_parse_hhmmss_timestamp_value() -> None:
    assert _audacity_timestamp("01:02:03.5", "f.txt", 1) == 3723.5


@pytest.mark.parametrize("value", ["", "abc", "1:2:3:4", "-1"])
def test_parse_invalid_timestamps(value: str) -> None:
    with pytest.raises(ValueError):
        _audacity_timestamp(value, "f.txt", 1)


def test_name_keys_match_across_extensions() -> None:
    assert _audacity_name_keys("Recording1.TXT") == {"recording1"}
    assert _audacity_name_keys("recording1.wav") == {"recording1"}
    assert _audacity_name_keys("recording1.wav.txt") == {"recording1.wav", "recording1"}
    assert _audacity_name_keys(None) == set()
    assert _audacity_name_keys("") == set()


def test_parse_label_filename_strips_media_extension() -> None:
    item = _parse_audacity_line("0.0\t1.0\tLabel", "recording1.wav.txt", 1)
    assert item.media_keys == ["recording1.wav", "recording1"]


def test_parse_rejects_wrong_column_counts() -> None:
    with pytest.raises(ValueError, match="expected 1, 2, 3, or 5 columns"):
        _parse_audacity_line("1.0\t2.0\t1000\t2000", "f.txt", 1)


def test_parse_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="end time precedes"):
        _parse_audacity_line("2.0\t1.0\tLabel", "f.txt", 1)


def test_parse_rejects_partial_frequency_bounds() -> None:
    with pytest.raises(ValueError, match="both low and high"):
        _parse_audacity_line("1.0\t2.0\t1000\t\tLabel", "f.txt", 1)


def test_parse_rejects_inverted_frequency_bounds() -> None:
    with pytest.raises(ValueError, match="high frequency precedes"):
        _parse_audacity_line("1.0\t2.0\t3400\t1200\tLabel", "f.txt", 1)
