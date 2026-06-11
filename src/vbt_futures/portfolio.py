"""Public-facing wrapper around the simulator output."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .records import FUTURES_ORDER_DT
from .spec import FuturesSpec


@dataclass(frozen=True)
class FuturesPortfolio:
    """Immutable post-simulation view of one backtest run."""

    close: pd.DataFrame
    specs: tuple[FuturesSpec, ...]
    init_cash: float
    freq: str | pd.Timedelta | None
    bars_per_year: float
    trading_days_per_year: int
    _order_records: np.ndarray
    _cash: np.ndarray
    _position: np.ndarray
    _margin_locked: np.ndarray

    # ---------- derived scalars (cached) ----------
    @cached_property
    def cash(self) -> pd.Series:
        """Available cash at the end of each bar (excludes locked margin)."""
        return pd.Series(self._cash, index=self.close.index, name="cash")

    @cached_property
    def position(self) -> pd.DataFrame:
        """Net position per contract at the end of each bar (+N long, -N short)."""
        return pd.DataFrame(
            self._position, index=self.close.index, columns=self.close.columns,
        )

    @cached_property
    def margin_locked(self) -> pd.DataFrame:
        """Locked margin per contract at the end of each bar."""
        return pd.DataFrame(
            self._margin_locked, index=self.close.index, columns=self.close.columns,
        )

    @cached_property
    def equity(self) -> pd.Series:
        """Total account equity = cash + sum(margin_locked)."""
        eq = self._cash + self._margin_locked.sum(axis=1)
        return pd.Series(eq, index=self.close.index, name="equity")

    @cached_property
    def returns(self) -> pd.Series:
        """Bar-to-bar returns of equity (first bar is 0.0)."""
        prev = np.concatenate(([self._cash[0] + self._margin_locked[0].sum()],
                                self._cash + self._margin_locked.sum(axis=1)))[:-1]
        eq = self._cash + self._margin_locked.sum(axis=1)
        r = np.zeros_like(eq)
        nonzero = prev != 0.0
        r[nonzero] = (eq[nonzero] - prev[nonzero]) / prev[nonzero]
        return pd.Series(r, index=self.close.index, name="returns")

    @cached_property
    def drawdown(self) -> pd.Series:
        """Running drawdown as a non-positive fraction (0 at first bar)."""
        eq = self._cash + self._margin_locked.sum(axis=1)
        cummax = np.maximum.accumulate(eq)
        dd = np.where(cummax > 0.0, eq / cummax - 1.0, 0.0)
        return pd.Series(dd, index=self.close.index, name="drawdown")

    # ---------- orders + trades ----------
    @cached_property
    def orders(self) -> pd.DataFrame:
        """Order records as a DataFrame (id, col, idx, size, price, fees, margin, side, pnl).

        Adds ``symbol`` and ``timestamp`` columns derived from the portfolio's
        ``close.index`` for ease of inspection.
        """
        if len(self._order_records) == 0:
            return pd.DataFrame(
                columns=[
                    "id", "col", "idx", "size", "price", "fees",
                    "margin", "side", "pnl", "symbol", "timestamp",
                ],
            )
        df = pd.DataFrame(self._order_records)
        df["symbol"] = [self.specs[c].symbol for c in df["col"]]
        df["timestamp"] = [self.close.index[i] for i in df["idx"]]
        return df

    @cached_property
    def trades(self) -> pd.DataFrame:
        """Round-trip trades obtained by pairing OPEN* with the next CLOSE*
        (or LIQUIDATED) per column.
        """
        if len(self._order_records) == 0:
            return pd.DataFrame(
                columns=[
                    "col", "symbol", "entry_time", "entry_price", "exit_time",
                    "exit_price", "size", "pnl", "fees", "duration_bars",
                    "is_liquidated",
                ],
            )
        rows: list[dict] = []
        OPEN_SIDES = {0, 2}    # OPEN_LONG, OPEN_SHORT
        CLOSE_SIDES = {1, 3, 4}  # CLOSE_LONG, CLOSE_SHORT, LIQUIDATED
        for col in range(len(self.specs)):
            col_mask = self._order_records["col"] == col
            col_orders = self._order_records[col_mask]
            open_rec: np.ndarray | None = None
            for rec in col_orders:
                side = int(rec["side"])
                if side in OPEN_SIDES:
                    open_rec = rec
                elif side in CLOSE_SIDES and open_rec is not None:
                    rows.append(
                        {
                            "col": col,
                            "symbol": self.specs[col].symbol,
                            "entry_time": self.close.index[int(open_rec["idx"])],
                            "entry_price": float(open_rec["price"]),
                            "exit_time": self.close.index[int(rec["idx"])],
                            "exit_price": float(rec["price"]),
                            "size": float(open_rec["size"]),
                            "pnl": float(rec["pnl"]),
                            "fees": float(open_rec["fees"]) + float(rec["fees"]),
                            "duration_bars": int(rec["idx"]) - int(open_rec["idx"]),
                            "is_liquidated": side == 4,
                        },
                    )
                    open_rec = None
        return pd.DataFrame(rows)
