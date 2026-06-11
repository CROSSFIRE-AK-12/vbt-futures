# vbt-futures

> **Backtrader-style futures backtesting on top of vectorbt's numba framework.**

`vbt-futures` is a Python library that gives you backtrader's **futures semantics** (margin, contract multiplier, mark-to-market, multi-side, forced liquidation) at vectorbt's **numba-level performance**. The simulator itself is a custom `@njit` routine that doesn't depend on vbt at runtime.

## Install

```bash
pip install -e ".[dev]"   # from this directory
```

## Quick Start

```python
import pandas as pd
import vbt_futures as vbtf

# Daily close prices for 3 contracts (rows: dates, columns: symbols)
close = pd.DataFrame(
    [[100.0, 200.0, 50.0], [101.0, 199.0, 51.0], ...],
    columns=["RB", "HC", "I"],
    index=pd.bdate_range("2024-01-02", periods=N),
)

# Boolean signal DataFrames (same shape as close)
long_entries  = pd.DataFrame(...)
long_exits    = pd.DataFrame(...)
short_entries = pd.DataFrame(...)
short_exits   = pd.DataFrame(...)

pf = vbtf.from_signals(
    close=close,
    long_entries=long_entries, long_exits=long_exits,
    short_entries=short_entries, short_exits=short_exits,
    specs=[
        vbtf.FuturesSpec("RB", mult=10.0,  margin_rate=0.10, fees=2e-4),
        vbtf.FuturesSpec("HC", mult=10.0,  margin_rate=0.10, fees=2e-4),
        vbtf.FuturesSpec("I",  mult=100.0, margin_rate=0.12, fees=2e-4),
    ],
    size=1.0,
    init_cash=200_000.0,
    freq="1D",
)

pf.cash              # pd.Series of available cash (excludes locked margin)
pf.position          # pd.DataFrame of net positions per contract
pf.margin_locked     # pd.DataFrame of locked margin per contract
pf.equity            # pd.Series = cash + sum(margin_locked)
pf.returns           # pd.Series of per-bar returns
pf.drawdown          # pd.Series of running drawdown (≤ 0)
pf.orders            # pd.DataFrame of all order records
pf.trades            # pd.DataFrame of round-trip trades
pf.stats()           # pd.Series with 22 summary metrics
pf.plot()            # plotly Figure (two panels: price+markers, equity+drawdown)
pf.to_vbt_orders()   # convert to vbt-style order DataFrame (for interop)
```

## API Reference

### `from_signals(close, *, long_entries, long_exits, short_entries, short_exits, specs, size, init_cash, freq, bars_per_year, trading_days_per_year, margin_mode, margin_lookback, margin_z_score, sizing_mode, sizing_kwargs) -> FuturesPortfolio`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `close` | `pd.DataFrame (T, N)` | required | Closing prices; rows = bars, columns = contracts |
| `long_entries` / `long_exits` / `short_entries` / `short_exits` | `pd.DataFrame (T, N)` bool | `None` | Entry/exit signal DataFrames. `None` ⇒ all False. |
| `specs` | `list[FuturesSpec]` | required | Per-contract specs; `len(specs) == close.shape[1]` |
| `size` | float / array / `DataFrame` | `1.0` | Base number of lots per entry. Direction is given by the signal. Must be `> 0`. |
| `init_cash` | float | `100_000` | Initial account cash (excluding margin). |
| `freq` | str / `pd.Timedelta` | `None` | pandas frequency string (informational). |
| `bars_per_year` | float | `None` | Override annualisation factor. `None` ⇒ auto-infer from `close.index`. |
| `trading_days_per_year` | int | `252` | Multiplier used when auto-inferring `bars_per_year`. Use 365 for crypto. |
| `margin_mode` | `"gross"` / `"portfolio"` | `"gross"` | `"gross"`: each contract's margin is `|pos|·close·mult·margin_rate`. `"portfolio"`: applies a covariance-based diversification discount. |
| `margin_lookback` | int | `60` | Number of bars to use for the rolling covariance matrix in `"portfolio"` mode. |
| `margin_z_score` | float | `1.65` | One-sided normal quantile for portfolio VaR (1.65 ≈ 95%). |
| `sizing_mode` | `"fixed"` / `"equity_proportional"` / `"anti_martingale"` | `"fixed"` | Position sizing rule (see [Sizing Modes](#sizing-modes) below). |
| `sizing_kwargs` | `dict` | `None` | Extra params for the sizing mode (e.g. `trigger_pnl`, `max_size`). |

### `FuturesSpec` (`@dataclass(frozen=True)`)

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | required | E.g. `"RB"` |
| `mult` | float | required | Contract multiplier (e.g. 10 = 10 tons per lot) |
| `margin_rate` | float | required | Fraction of notional charged as margin (e.g. 0.10 = 10%) |
| `fees` | float | `0.0` | Commission as fraction of order notional |
| `fixed_fees` | float | `0.0` | Per-lot fixed commission in account currency |
| `slippage` | float | `0.0` | Price penalty on fills (e.g. 0.01 = 1%) |
| `tick_size` | float | `0.01` | Minimum price increment (informational only) |
| `flat_conflict` | `"long"` / `"short"` / `"skip"` | `"skip"` | What to do when both long & short entry fire while flat |

### Sizing Modes

| Mode | Behaviour |
|---|---|
| `"fixed"` (default) | Every entry uses `size` directly. |
| `"equity_proportional"` | `size` scales linearly with running equity: `size × equity_so_far / init_cash`. As the account grows, position size grows; as it shrinks, size shrinks. |
| `"anti_martingale"` | Each unit of cum PnL above `trigger_pnl` adds one base lot to size, capped at `max_size`. Does NOT shrink size on losses (anti-martingale). |

```python
# Anti-martingale: every 1000 RMB of profit adds 1 lot, max 5 lots
pf = vbtf.from_signals(
    ..., sizing_mode="anti_martingale", size=1.0,
    sizing_kwargs={"trigger_pnl": 1000.0, "max_size": 5.0},
)
```

### Portfolio Margin

`margin_mode="portfolio"` re-estimates the per-bar margin requirement using a
rolling N×N covariance matrix.  The discount factor is:

```
portfolio_dollar_VaR  = sqrt(p^T Σ p)            where p = |position[col]| × close × mult
sum_individual_VaR     = Σ |p_col| × std_col
discount               = portfolio_dollar_VaR / sum_individual_VaR
margin_locked[col]     = gross_margin[col] × discount
```

Properties:
- **Perfect positive correlation** → discount ≈ 1.0 (no diversification benefit)
- **Zero correlation** → discount < 1.0 (full diversification)
- **Auto-adapts**: discount changes daily as the rolling covariance evolves

```python
pf = vbtf.from_signals(
    ..., margin_mode="portfolio", margin_lookback=60,
)
```

In the demo (`examples/demo_synthetic.py`), the 3-contract portfolio sees a
~28% margin reduction vs gross because contracts 0/1 are highly correlated
with each other but uncorrelated with contract 2.

### Signal Processing Rules (per bar, per column)

1. **Pass 1** — handle an existing position:
   - Long + `short_entry` → close long + open short (**reversal**, 2 records)
   - Long + `long_exit` → close long (1 record)
   - Short + `long_entry` → close short + open long (**reversal**)
   - Short + `short_exit` → close short
2. **Pass 2** — handle an entry from flat (position became 0 in Pass 1):
   - Both long & short entry True → `flat_conflict` policy (default skip)
   - `long_entry` → open long
   - `short_entry` → open short
3. **Same-bar close+entry** is treated as a **refresh** of the position (close then re-open).

## Known Limitations

1. **Gap P&L is fully captured**, but three approximations remain:
   - Mark-to-market uses **close** rather than the exchange's **settlement price**
   - **Liquidation** fires at K-line close, not the gap-open price
   - All triggers read **close** only; we don't process high/low
2. **No continuous-contract rollover** — users must concatenate contracts themselves
3. **Annualisation default is "trading session" inferred** from `close.index` (median bars-per-day × 252). Override via `bars_per_year` / `trading_days_per_year`.
4. **No SPAN-style portfolio margin**
5. **No long+short lock** (single position per contract at a time)

## Development

```bash
# Run all tests with 100% coverage
pytest

# Run the synthetic demo
python examples/demo_synthetic.py

# Run the performance benchmark (with NUMBA JIT enabled)
python benchmarks/bench_simulator.py
```

Coverage: **100% lines, 100% branches** (`pytest --cov-branch`).

## License

MIT
