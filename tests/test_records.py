"""Tests for src/vbt_futures/records.py."""

import numpy as np
import pytest

from vbt_futures.records import FUTURES_ORDER_DT, futures_order_dt, make_empty_records


def test_futures_order_dt_singleton() -> None:
    """The factory is idempotent and returns the same singleton dtype."""
    a = futures_order_dt()
    b = futures_order_dt()
    assert a is b
    assert a is FUTURES_ORDER_DT


def test_dtype_field_names() -> None:
    names = FUTURES_ORDER_DT.names
    assert names is not None
    assert tuple(names) == (
        "id",
        "col",
        "idx",
        "size",
        "price",
        "fees",
        "margin",
        "side",
        "pnl",
    )


def test_dtype_dtypes() -> None:
    expected = {
        "id": np.int64,
        "col": np.int64,
        "idx": np.int64,
        "size": np.float64,
        "price": np.float64,
        "fees": np.float64,
        "margin": np.float64,
        "side": np.int64,
        "pnl": np.float64,
    }
    for name, expected_dt in expected.items():
        actual_dt = FUTURES_ORDER_DT.fields[name][0]
        assert actual_dt == expected_dt, f"{name}: expected {expected_dt}, got {actual_dt}"


def test_align_is_nonzero() -> None:
    """align=True was requested. In numpy 2.x, the attribute is `alignment`
    (int, >0 means aligned; value >1 means explicit padding)."""
    # Newer numpy exposes `.alignment`; older/other builds may expose `.isaligned`
    if hasattr(FUTURES_ORDER_DT, "alignment"):
        assert FUTURES_ORDER_DT.alignment > 0
    elif hasattr(FUTURES_ORDER_DT, "isaligned"):
        assert FUTURES_ORDER_DT.isaligned is True
    else:  # pragma: no cover -- numpy API changed
        pytest.skip("numpy dtype has neither `.alignment` nor `.isaligned`")


def test_make_empty_records_returns_zero_length_array() -> None:
    arr = make_empty_records(0)
    assert len(arr) == 0
    assert arr.dtype == FUTURES_ORDER_DT


def test_make_empty_records_returns_n_length_array() -> None:
    arr = make_empty_records(7)
    assert len(arr) == 7
    assert arr.dtype == FUTURES_ORDER_DT


def test_field_writable() -> None:
    """Records must be writable so simulator can fill them in-place."""
    arr = make_empty_records(1)
    arr[0]["id"] = 42
    arr[0]["col"] = 0
    arr[0]["size"] = 1.5
    arr[0]["price"] = 100.0
    assert arr[0]["id"] == 42
    assert arr[0]["col"] == 0
    assert arr[0]["size"] == 1.5
    assert arr[0]["price"] == 100.0
