"""Tests for src/vbt_futures/enums.py."""

from vbt_futures.enums import (
    CLOSE_LONG,
    CLOSE_SHORT,
    FLAT_CONFLICT_CODE,
    LIQUIDATED,
    OPEN_LONG,
    OPEN_SHORT,
)


def test_open_long_is_zero() -> None:
    assert OPEN_LONG == 0


def test_close_long_is_one() -> None:
    assert CLOSE_LONG == 1


def test_open_short_is_two() -> None:
    assert OPEN_SHORT == 2


def test_close_short_is_three() -> None:
    assert CLOSE_SHORT == 3


def test_liquidated_is_four() -> None:
    assert LIQUIDATED == 4


def test_side_codes_are_distinct() -> None:
    codes = {OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT, LIQUIDATED}
    assert len(codes) == 5


def test_flat_conflict_long_is_zero() -> None:
    assert FLAT_CONFLICT_CODE["long"] == 0


def test_flat_conflict_short_is_one() -> None:
    assert FLAT_CONFLICT_CODE["short"] == 1


def test_flat_conflict_skip_is_two() -> None:
    assert FLAT_CONFLICT_CODE["skip"] == 2


def test_flat_conflict_has_exactly_three_keys() -> None:
    assert set(FLAT_CONFLICT_CODE.keys()) == {"long", "short", "skip"}
