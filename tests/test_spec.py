"""Tests for src/vbt_futures/spec.py."""

import pytest

from vbt_futures.spec import FuturesSpec


def test_minimal_construction() -> None:
    spec = FuturesSpec(symbol="RB", mult=10.0, margin_rate=0.10)
    assert spec.symbol == "RB"
    assert spec.mult == 10.0
    assert spec.margin_rate == 0.10
    assert spec.fees == 0.0
    assert spec.fixed_fees == 0.0
    assert spec.slippage == 0.0
    assert spec.tick_size == 0.01
    assert spec.flat_conflict == "skip"


def test_full_construction() -> None:
    spec = FuturesSpec(
        symbol="I",
        mult=100.0,
        margin_rate=0.12,
        fees=2e-4,
        fixed_fees=3.0,
        slippage=1e-4,
        tick_size=0.5,
        flat_conflict="long",
    )
    assert spec.fees == 2e-4
    assert spec.fixed_fees == 3.0
    assert spec.slippage == 1e-4
    assert spec.tick_size == 0.5
    assert spec.flat_conflict == "long"


def test_is_frozen() -> None:
    spec = FuturesSpec(symbol="RB", mult=10.0, margin_rate=0.10)
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError is a subclass
        spec.mult = 20.0  # type: ignore[misc]


def test_equality() -> None:
    a = FuturesSpec(symbol="RB", mult=10.0, margin_rate=0.10)
    b = FuturesSpec(symbol="RB", mult=10.0, margin_rate=0.10)
    assert a == b


def test_inequality_on_mult() -> None:
    a = FuturesSpec(symbol="RB", mult=10.0, margin_rate=0.10)
    b = FuturesSpec(symbol="RB", mult=20.0, margin_rate=0.10)
    assert a != b


def test_inequality_on_symbol() -> None:
    a = FuturesSpec(symbol="RB", mult=10.0, margin_rate=0.10)
    b = FuturesSpec(symbol="HC", mult=10.0, margin_rate=0.10)
    assert a != b


def test_repr_includes_symbol() -> None:
    spec = FuturesSpec(symbol="RB", mult=10.0, margin_rate=0.10)
    r = repr(spec)
    assert "RB" in r
