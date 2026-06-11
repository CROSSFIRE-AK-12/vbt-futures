"""Tests for src/vbt_futures/utils.py::_validate_inputs.

One test per ValueError trigger in spec §7.1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbt_futures.spec import FuturesSpec
from vbt_futures.utils import _validate_inputs


@pytest.fixture
def good_close() -> pd.DataFrame:
    return pd.DataFrame(
        [[100.0, 200.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )


@pytest.fixture
def good_specs() -> list[FuturesSpec]:
    return [
        FuturesSpec(symbol="RB", mult=10.0, margin_rate=0.10),
        FuturesSpec(symbol="HC", mult=10.0, margin_rate=0.10),
    ]


# ---------- close not a DataFrame ----------
def test_close_not_dataframe_raises() -> None:
    with pytest.raises(ValueError, match="close 必须是 pd.DataFrame"):
        _validate_inputs(
            close="not a df",  # type: ignore[arg-type]
            long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("X", 1.0, 0.1)],
            size=1.0,
            init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- len(specs) mismatch ----------
def test_specs_length_mismatch_raises(good_close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="不匹配 close 的列数"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10)],  # only 1, close has 2 cols
            size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- margin_rate <= 0 ----------
def test_margin_rate_non_positive_raises(good_close: pd.DataFrame) -> None:
    specs = [
        FuturesSpec("RB", 10.0, 0.10),
        FuturesSpec("HC", 10.0, -0.05),  # bad
    ]
    with pytest.raises(ValueError, match="margin_rate 必须 > 0"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=specs, size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- mult <= 0 ----------
def test_mult_non_positive_raises(good_close: pd.DataFrame) -> None:
    specs = [
        FuturesSpec("RB", 0.0, 0.10),  # bad
        FuturesSpec("HC", 10.0, 0.10),
    ]
    with pytest.raises(ValueError, match="mult 必须 > 0"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=specs, size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- init_cash <= 0 ----------
def test_init_cash_non_positive_raises(good_close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="init_cash 必须 > 0"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=1.0, init_cash=-1.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- signal shape mismatch ----------
def test_long_entries_shape_mismatch_raises(good_close: pd.DataFrame) -> None:
    bad = pd.DataFrame([[True]], columns=["RB"], index=good_close.index)
    with pytest.raises(ValueError, match="long_entries 形状 .* 不匹配 close"):
        _validate_inputs(
            close=good_close, long_entries=bad, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- signal non-bool dtype ----------
def test_long_entries_non_bool_dtype_raises(good_close: pd.DataFrame) -> None:
    bad = pd.DataFrame(
        np.zeros(good_close.shape, dtype=np.int64),
        columns=good_close.columns, index=good_close.index,
    )
    with pytest.raises(ValueError, match="long_entries 必须 bool 类型"):
        _validate_inputs(
            close=good_close, long_entries=bad, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- size <= 0 ----------
def test_size_non_positive_raises(good_close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="size 含非正数"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=-2.0, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


def test_size_array_non_positive_raises(good_close: pd.DataFrame) -> None:
    bad_size = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match="size 含非正数"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=bad_size, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- close negative values ----------
def test_close_negative_value_raises() -> None:
    close = pd.DataFrame(
        [[100.0, -5.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    with pytest.raises(ValueError, match="close 含负值"):
        _validate_inputs(
            close=close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- flat_conflict invalid ----------
def test_flat_conflict_invalid_raises(good_close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="flat_conflict 必须是 'long'|'short'|'skip'"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[
                FuturesSpec("RB", 10.0, 0.10, flat_conflict="random"),
                FuturesSpec("HC", 10.0, 0.10),
            ],
            size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- freq unparseable ----------
def test_freq_unparseable_raises(good_close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="freq='banana' 不能解析为 pd.Timedelta"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=1.0, init_cash=10_000.0,
            freq="banana", bars_per_year=None, trading_days_per_year=252,
        )


# ---------- bars_per_year <= 0 ----------
def test_bars_per_year_non_positive_raises(good_close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="bars_per_year 必须 > 0"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=-10.0, trading_days_per_year=252,
        )


# ---------- trading_days_per_year <= 0 ----------
def test_trading_days_per_year_non_positive_raises(good_close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="trading_days_per_year 必须 > 0"):
        _validate_inputs(
            close=good_close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=1.0, init_cash=10_000.0,
            freq="1D", bars_per_year=252.0, trading_days_per_year=0,
        )


# ---------- auto-infer impossible ----------
def test_non_datetime_index_no_bars_per_year_raises() -> None:
    close = pd.DataFrame(
        [[100.0, 200.0]],
        columns=["RB", "HC"],
        index=[0],  # plain RangeIndex
    )
    with pytest.raises(ValueError, match="无法推断 bars_per_year"):
        _validate_inputs(
            close=close, long_entries=None, long_exits=None,
            short_entries=None, short_exits=None,
            specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
            size=1.0, init_cash=10_000.0,
            freq=None, bars_per_year=None, trading_days_per_year=252,
        )


# ---------- happy path: all OK ----------
def test_valid_inputs_pass(good_close: pd.DataFrame) -> None:
    """Sanity check: no exception is raised on a well-formed input."""
    _validate_inputs(
        close=good_close, long_entries=None, long_exits=None,
        short_entries=None, short_exits=None,
        specs=[FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.10)],
        size=1.0, init_cash=10_000.0,
        freq="1D", bars_per_year=None, trading_days_per_year=252,
    )
