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
    _entry_mask_long: np.ndarray = None  # type: ignore[assignment]
    _entry_mask_long_exit: np.ndarray = None  # type: ignore[assignment]
    _entry_mask_short: np.ndarray = None  # type: ignore[assignment]
    _entry_mask_short_exit: np.ndarray = None  # type: ignore[assignment]

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

    # ---------- summary stats ----------
    def stats(self) -> pd.Series:
        """One-row summary of the run (22 fields per spec §6.5).

        Sharpe / Sortino / Annualized return are computed from the
        per-bar return series using ``bars_per_year`` as the
        annualisation factor (no vbt dependency).
        """
        eq = self.equity
        ret = self.returns.values
        final_equity = float(eq.iloc[-1])
        total_return = (final_equity / self.init_cash) - 1.0
        bpy = float(self.bars_per_year)

        # Annualised return: geometric for long horizons is fine to
        # approximate as bpy * mean(ret) when returns are small, but
        # compute properly for accuracy.
        if len(ret) > 1 and (1.0 + ret[1:]).all() > 0.0:
            period_return = float(np.prod(1.0 + ret[1:]) - 1.0)
        else:
            period_return = float(ret.sum())  # pragma: no cover
        if period_return > -1.0 and bpy > 0.0:
            ann_ret = (1.0 + period_return) ** (bpy / max(len(ret) - 1, 1)) - 1.0
        else:
            ann_ret = float("nan")  # pragma: no cover

        # Sharpe: mean / std * sqrt(bpy)
        r = ret[1:]  # skip the 0.0 first bar
        std = float(np.std(r, ddof=0))
        mean = float(np.mean(r))
        sharpe = (mean / std) * np.sqrt(bpy) if std > 0.0 else float("nan")

        # Sortino: mean / downside_std * sqrt(bpy)
        downside = r[r < 0.0]
        down_std = float(np.std(downside, ddof=0)) if len(downside) > 0 else 0.0
        sortino = (mean / down_std) * np.sqrt(bpy) if down_std > 0.0 else float("nan")

        t = self.trades
        if len(t) == 0:
            win_mask = pd.Series([], dtype=bool)
            loss_mask = pd.Series([], dtype=bool)
        else:
            win_mask = t["pnl"] > 0
            loss_mask = t["pnl"] < 0
        total_profit = float(t.loc[win_mask, "pnl"].sum()) if win_mask.any() else 0.0
        total_loss = float(t.loc[loss_mask, "pnl"].sum()) if loss_mask.any() else 0.0
        profit_factor = (total_profit / -total_loss) if total_loss != 0.0 else float("nan")
        avg_win = float(t.loc[win_mask, "pnl"].mean()) if win_mask.any() else float("nan")
        avg_loss = float(t.loc[loss_mask, "pnl"].mean()) if loss_mask.any() else float("nan")
        win_loss_ratio = (avg_win / -avg_loss) if (avg_loss != 0.0 and avg_loss == avg_loss) else float("nan")

        return pd.Series(
            {
                "Start": eq.index[0],
                "End": eq.index[-1],
                "Period": len(eq),
                "Init Cash": self.init_cash,
                "Final Equity": final_equity,
                "Total Return [%]": total_return * 100.0,
                "Annualized Return [%]": ann_ret * 100.0,
                "Sharpe Ratio": float(sharpe),
                "Sortino Ratio": float(sortino),
                "Max Drawdown [%]": float(self.drawdown.min() * 100.0),
                "Total Trades": len(t),
                "Win Rate [%]": (win_mask.sum() / len(t) * 100.0) if len(t) > 0 else float("nan"),
                "Profit Factor": profit_factor,
                "Avg Trade PnL": float(t["pnl"].mean()) if len(t) > 0 else float("nan"),
                "Avg Win": avg_win,
                "Avg Loss": avg_loss,
                "Win/Loss Ratio": win_loss_ratio,
                "Max Position": float(self.position.abs().values.max()),
                "Total Fees": float(self.orders["fees"].sum()) if len(self.orders) > 0 else 0.0,
                "Liquidations": int(t["is_liquidated"].sum()) if len(t) > 0 else 0,
                "Bars Per Year": self.bars_per_year,
                "Trading Days Per Year": self.trading_days_per_year,
            },
        )

    # ---------- plotting ----------
    def plot(self):
        """Return a plotly Figure with two stacked panels:
        top = close prices (one line per contract) + order markers;
        bottom = equity curve + drawdown.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3], vertical_spacing=0.05,
        )
        # Panel 1: close prices.
        for col_name in self.close.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.close.index, y=self.close[col_name],
                    mode="lines", name=f"{col_name} close",
                ),
                row=1, col=1,
            )
        # Order markers.
        if len(self.orders) > 0:
            for side_val, label, marker_sym, color in [
                (0, "open long", "triangle-up", "green"),
                (1, "close long", "triangle-down", "red"),
                (2, "open short", "triangle-down", "orange"),
                (3, "close short", "triangle-up", "blue"),
                (4, "liquidated", "x", "black"),
            ]:
                mask = self.orders["side"] == side_val
                if mask.any():
                    sub = self.orders[mask]
                    fig.add_trace(
                        go.Scatter(
                            x=sub["timestamp"],
                            y=[self.close.iloc[int(i), int(c)] for i, c in zip(sub["idx"], sub["col"])],
                            mode="markers",
                            name=label,
                            marker=dict(symbol=marker_sym, size=10, color=color),
                        ),
                        row=1, col=1,
                    )
        # Panel 2: equity + drawdown.
        fig.add_trace(
            go.Scatter(x=self.equity.index, y=self.equity.values, mode="lines", name="equity"),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.drawdown.index, y=self.drawdown.values * 100.0,
                mode="lines", name="drawdown %", fill="tozeroy",
                line=dict(color="red"),
            ),
            row=2, col=1,
        )
        fig.update_layout(height=700, title_text="vbt-futures backtest")
        return fig

    # ---------- sizing visualization ----------
    def plot_sizing(
        self,
        base_size: float | np.ndarray | pd.DataFrame = 1.0,
        sizing_mode: str = "fixed",
        sizing_kwargs: dict | None = None,
    ):
        """Two-panel plot showing equity AND size BOTH over TIME.

        Both panels share the same X axis (time), so you can directly see
        how size scales with equity across the backtest timeline.

        - Top panel: equity curve with orange triangle markers at every
          entry bar.
        - Bottom panel: size (lots) at every entry bar, with the
          previous size carried forward as a horizontal step.  When
          ``sizing_mode="equity_proportional"`` and equity is dropping,
          you'll see the size step *downward* over time.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from .sizing import resolve_size

        T, N = self.close.shape
        if sizing_mode == "fixed":
            size_arr = np.full((T, N), float(base_size), dtype=np.float64)
        else:
            size_arr = resolve_size(
                sizing_mode, base_size,
                self._entry_mask_long, self._entry_mask_long_exit,
                self._entry_mask_short, self._entry_mask_short_exit,
                self.close.values.astype(np.float64),
                float(self.init_cash), sizing_kwargs,
            )

        # Find entry bars (any column transition 0 -> non-zero).
        pos = self._position
        any_pos = (pos != 0).any(axis=1)
        prev_any = np.concatenate(([False], any_pos[:-1]))
        entry_bars = np.where(any_pos & ~prev_any)[0]

        # Per-bar size (mean across columns, since all cols are equal in v1).
        size_per_bar = size_arr.mean(axis=1)

        # Build a step series that holds the last entry's size until next entry.
        size_step = np.zeros(T, dtype=np.float64)
        if len(entry_bars) > 0:
            current = size_per_bar[entry_bars[0]]
            for t in range(T):
                if t in entry_bars:
                    current = size_per_bar[t]
                size_step[t] = current

        eq_arr = self.equity.values

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.5], vertical_spacing=0.10,
            subplot_titles=(
                "Equity curve over time (orange = entry bar)",
                "Size on entry bars over time (step line)",
            ),
        )

        # ---- Top: equity curve ----
        fig.add_trace(
            go.Scatter(
                x=self.equity.index, y=eq_arr,
                mode="lines", name="equity",
                line=dict(color="steelblue", width=2),
            ),
            row=1, col=1,
        )
        if len(entry_bars) > 0:
            fig.add_trace(
                go.Scatter(
                    x=self.equity.index[entry_bars], y=eq_arr[entry_bars],
                    mode="markers", name="entry bar",
                    marker=dict(symbol="triangle-up", size=10, color="orange"),
                ),
                row=1, col=1,
            )

        # ---- Bottom: size step line ----
        fig.add_trace(
            go.Scatter(
                x=self.equity.index, y=size_step,
                mode="lines", name="size (held until next entry)",
                line=dict(color="darkgreen", width=2, shape="hv"),
            ),
            row=2, col=1,
        )
        if len(entry_bars) > 0:
            fig.add_trace(
                go.Scatter(
                    x=self.equity.index[entry_bars],
                    y=size_per_bar[entry_bars],
                    mode="markers", name="size at entry",
                    marker=dict(size=10, color="darkgreen", symbol="circle"),
                ),
                row=2, col=1,
            )
        # Reference line: base size.
        fig.add_hline(
            y=float(np.mean(size_arr[0, :])) if T > 0 else 0.0,
            line_dash="dash", line_color="gray", opacity=0.5,
            annotation_text="base_size",
            annotation_position="right",
            row=2, col=1,
        )

        fig.update_yaxes(title_text="equity", row=1, col=1)
        fig.update_yaxes(title_text="size (lots)", row=2, col=1)
        fig.update_layout(
            height=800,
            title_text=f"vbt-futures: sizing mode = {sizing_mode}  (size step follows equity)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ---------- vbt interop ----------
    def to_vbt_orders(self) -> pd.DataFrame:
        """Convert internal ``futures_order_dt`` to vbt-style order DataFrame.

        vbt's order format uses ``side`` 0=Buy, 1=Sell.  We map
        OPEN_LONG/CLOSE_SHORT -> 0, and OPEN_SHORT/CLOSE_LONG -> 1.
        LIQUIDATED is mapped to Sell (1).
        """
        if len(self.orders) == 0:
            return pd.DataFrame(columns=["id", "col", "idx", "size", "price", "fees", "side"])
        df = self.orders[["id", "col", "idx", "size", "price", "fees", "side"]].copy()
        vbt_side = df["side"].map(
            {0: 0, 1: 1, 2: 1, 3: 0, 4: 1},
        )
        df["side"] = vbt_side
        return df
