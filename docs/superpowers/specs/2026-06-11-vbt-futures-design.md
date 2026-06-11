# vbt-futures 设计文档

| 字段 | 值 |
|---|---|
| **项目** | vbt-futures |
| **作者** | Claude (与用户对齐) |
| **日期** | 2026-06-11 |
| **状态** | Draft → 待用户复核 |
| **依赖文档** | `backtrader-master/backtrader/comminfo.py`（期货语义参照）<br>`vectorbt-master/vectorbt/portfolio/nb.py`（numba 框架参照） |

---

## 1. 背景与目标

### 1.1 问题陈述

- **backtrader** 原生支持期货（保证金、合约乘数、逐日盯市），但作者于 2019 年后停更，Python 3.13 兼容性勉强可用、性能受限于单线程 Python 主循环。
- **vectorbt** 提供高性能 numba 回测框架，但**核心假设是股票模型**——`buy_nb()` 直接按"全额名义价值"扣现金（见 `vectorbt/portfolio/nb.py:148`：`req_cash = adj_size * adj_price`），没有 `margin` 与 `mult` 概念。
- 因此 vectorbt 不能直接用于期货回测。

### 1.2 目标

实现一个新模块 **vbt-futures**：使用 vectorbt 的 numba 基础设施（数据结构、record dtype、returns_accessor），**自己写一个 njit simulator** 实现 backtrader 风格的期货语义，对外提供与 vbt 类似的信号驱动 API。

### 1.3 非目标

- 不实现期权希腊字母
- 不实现移仓换月（用户自己拼连续合约）
- 不实现 SPAN 组合保证金
- 不替代 backtrader / vectorbt——只针对"在 vbt 框架里跑期货"这一个场景

---

## 2. 需求决策（用户已确认）

| # | 决策点 | 选项 |
|---|---|---|
| 1 | 范围 | 多合约组合（N 列独立品种，共享现金账户） |
| 2 | 保证金模型 | 按保证金率：`margin = price × mult × margin_rate`，每根 K 线动态重算 |
| 3 | API 风格 | `from_signals` 风格（DataFrame 输入输出） |
| 4 | 方向 | 多空双向 + 反手，单边持仓（不锁仓） |
| 5 | 仓位大小 | 固定手数（`size=N` 表示 N 手） |
| 6 | 风控 | 保证金不足 → 拒单（不留 record）；总权益 ≤ 0 → 下根 K 线强平所有持仓 |
| 7 | 输出 | 自定义 `FuturesPortfolio` 对象 |
| 8 | 实现方案 | 方案 A：独立 njit simulator，复用 vbt 数据结构和 returns_accessor |
| 9 | Demo 数据 | 合成 OHLC（3 个虚拟品种，2 年日线，固定 seed） |
| 10 | 成交价 | 信号 K 线的 `close[t]`，在出现信号的 K 线上成交 |
| 11 | 同 bar long+short entry 同 True | 持仓时按反手规则处理；空仓时跳过（默认 `flat_conflict="skip"`） |

---

## 3. 架构

### 3.1 目录结构

```
vbt-futures/
├── pyproject.toml
├── README.md
├── pytest.ini
├── src/vbt_futures/
│   ├── __init__.py            # 公开 API
│   ├── spec.py                # FuturesSpec
│   ├── enums.py               # Side / FlatConflict 常量
│   ├── records.py             # futures_order_dt
│   ├── simulator.py           # ⭐ @njit simulate_futures_nb
│   ├── portfolio.py           # FuturesPortfolio
│   └── utils.py               # 输入校验 / 信号广播
├── examples/
│   └── demo_synthetic.py
├── benchmarks/
│   └── bench_simulator.py
├── tests/
│   ├── conftest.py
│   ├── test_simulator.py
│   ├── test_margin.py
│   ├── test_liquidation.py
│   ├── test_signals.py
│   ├── test_portfolio.py
│   ├── test_derived.py
│   ├── test_validation.py
│   ├── test_freq.py
│   └── test_vs_backtrader.py  # marker=slow
└── docs/superpowers/specs/
    └── 2026-06-11-vbt-futures-design.md
```

### 3.2 模块依赖图（单向）

```
public API (from_signals)
       │
       ▼
   simulator.py  ──→  spec.py  +  enums.py  +  records.py
       │                                  ▲
       │                                  │ 借用
       ▼                              vectorbt.portfolio.enums
   portfolio.py  ──→  vectorbt.returns_accessor
```

### 3.3 设计原则

- `simulator.py` 是**唯一**含 `@njit` 的模块，单独可测可换
- `portfolio.py` 是 Python 包装层，不被 numba 约束，可自由用 pandas / plotly
- `FuturesSpec` 用 `@dataclass(frozen=True)`，遵循项目不可变规则
- 所有文件 ≤ 800 行

---

## 4. 数据模型

### 4.1 FuturesSpec（用户面）

```python
@dataclass(frozen=True)
class FuturesSpec:
    symbol: str                       # "RB"
    mult: float                       # 合约乘数, e.g. 10
    margin_rate: float                # 保证金率, e.g. 0.10
    fees: float = 0.0                 # 成交金额%, e.g. 2e-4
    fixed_fees: float = 0.0           # 每手固定费 RMB
    slippage: float = 0.0             # 价格%
    tick_size: float = 0.01           # 仅记录, 不参与撮合
    flat_conflict: str = "skip"       # "long" | "short" | "skip"
```

### 4.2 simulator 输入（全 numpy）

| 名称 | shape | dtype | 含义 |
|---|---|---|---|
| `close` | (T, N) | float64 | 收盘价（盯市基准 + 成交价） |
| `long_entries` | (T, N) | bool | 开多信号 |
| `long_exits` | (T, N) | bool | 平多信号 |
| `short_entries` | (T, N) | bool | 开空信号 |
| `short_exits` | (T, N) | bool | 平空信号 |
| `size` | (T, N) | float64 | 每次开仓手数（>0） |
| `mult` | (N,) | float64 | 合约乘数 |
| `margin_rate` | (N,) | float64 | 保证金率 |
| `fees` | (N,) | float64 | 百分比手续费 |
| `fixed_fees` | (N,) | float64 | 每手固定费 |
| `slippage` | (N,) | float64 | 滑点 |
| `flat_conflict_code` | (N,) | int8 | 0=long / 1=short / 2=skip |
| `init_cash` | 标量 | float64 | 初始资金 |

### 4.3 simulator 输出

| 名称 | shape | dtype | 含义 |
|---|---|---|---|
| `order_records` | (M,) | `futures_order_dt` | 按发生顺序的成交记录，**M ≤ 2·T·N**，simulator 内部预分配 `2·T·N` 容量再按真实计数 M 切片返回 |
| `cash` | (T,) | float64 | 每根 K 线末可用现金（不含保证金） |
| `position` | (T, N) | float64 | 持仓手数（+多/-空/0） |
| `margin_locked` | (T, N) | float64 | 锁定保证金 |

衍生（FuturesPortfolio 层）：`equity[t] = cash[t] + sum(margin_locked[t, :])`

### 4.4 futures_order_dt

```python
futures_order_dt = np.dtype([
    ("id",     np.int64),     # 顺序 ID
    ("col",    np.int64),     # 列索引
    ("idx",    np.int64),     # K 线索引
    ("size",   np.float64),   # 手数（带符号: +多/-空）
    ("price",  np.float64),   # 成交价（含滑点）
    ("fees",   np.float64),   # 手续费
    ("margin", np.float64),   # 锁定/释放的保证金（开仓+，平仓-）
    ("side",   np.int64),     # 0=开多 / 1=平多 / 2=开空 / 3=平空 / 4=强平
    ("pnl",    np.float64),   # 平仓实现盈亏（开仓时 0）
], align=True)
```

预分配大小：`max_orders = 2 * T * N`（最坏情况每根 K 线每列都反手）。

---

## 5. 核心算法

### 5.1 信号处理规则（每列每根 K 线，**两阶段评估**）

```
Per col per bar:

═══ PASS 1 — 处理已有仓位 (exit / reversal) ═══

  if position > 0 (持多):
      if short_entry:           REVERSE_TO_SHORT (平多 + 开空); 跳过 PASS 2
      elif long_exit:           CLOSE long  (position 变 0); 继续 PASS 2
      else:                     hold; 跳过 PASS 2

  elif position < 0 (持空):
      if long_entry:            REVERSE_TO_LONG  (平空 + 开多); 跳过 PASS 2
      elif short_exit:          CLOSE short (position 变 0); 继续 PASS 2
      else:                     hold; 跳过 PASS 2

  else (position == 0):
      继续 PASS 2

═══ PASS 2 — 从空仓开仓 (仅当 PASS 1 后 position==0) ═══

  if long_entry AND short_entry 同 True:
      flat_conflict="long":     OPEN long
      flat_conflict="short":    OPEN short
      flat_conflict="skip":     保持空仓 (默认)
  elif long_entry:              OPEN long
  elif short_entry:             OPEN short
  else:                         保持空仓
```

**两阶段评估覆盖的关键场景**：

| 起始持仓 | 触发信号 | PASS 1 | PASS 2 | 最终 |
|---|---|---|---|---|
| 持多 | `long_exit` + `long_entry` | 平多 | 重开多 | 同向止盈再入 (2 笔) |
| 持多 | `short_entry` | 反手到空 | (跳过) | 多→空 (2 笔) |
| 持多 | `long_exit` | 平多 | flat 时无信号 | 空仓 (1 笔) |
| 持空 | `short_exit` + `short_entry` | 平空 | 重开空 | 同向止盈再入 (2 笔) |
| 空仓 | `long_entry` only | (无操作) | 开多 | 持多 (1 笔) |
| 空仓 | `long_entry` + `short_entry` | (无操作) | 默认 skip | 仍空仓 (0 笔) |

**反手关键性质**：
- 反手在 PASS 1 中产生，优先级高于普通 exit（信息量更大）
- 反手生成 2 笔 order record（先 CLOSE 后 OPEN）
- 共享同一根 K 线、同一成交价 `close[t]`
- 手续费分别计算

### 5.2 K 线主循环（每个 t 共 5 步）

```python
for t in range(T):
    # STEP 1: 盯市
    for col in range(N):
        if position[col] != 0 and not liquidated[col]:
            cash += position[col] * (close[t, col] - prev_close[col]) * mult[col]

    # STEP 2: 信号处理（按 §5.1 两阶段规则，每列独立）
    for col in range(N):
        if liquidated[col]: continue

        # --- PASS 1: 处理已有仓位 (exit / reversal) ---
        pass1 = decide_pass1(
            position[col], long_entries[t, col], long_exits[t, col],
            short_entries[t, col], short_exits[t, col],
        )
        # pass1 ∈ {HOLD, CLOSE, REVERSE_TO_LONG, REVERSE_TO_SHORT}

        if pass1 == REVERSE_TO_LONG:
            do_close(col, close[t, col])
            try_open(col, +size[t, col], close[t, col])
            continue                            # 反手后跳过 PASS 2
        if pass1 == REVERSE_TO_SHORT:
            do_close(col, close[t, col])
            try_open(col, -size[t, col], close[t, col])
            continue
        if pass1 == CLOSE:
            do_close(col, close[t, col])
            # 位置归零 → 落到 PASS 2

        # --- PASS 2: 仅当 PASS 1 后 position[col]==0 ---
        if position[col] != 0:
            continue
        pass2 = decide_pass2(
            long_entries[t, col], short_entries[t, col],
            flat_conflict_code[col],
        )
        # pass2 ∈ {HOLD, OPEN_LONG, OPEN_SHORT}
        if pass2 == OPEN_LONG:
            try_open(col, +size[t, col], close[t, col])
        elif pass2 == OPEN_SHORT:
            try_open(col, -size[t, col], close[t, col])

    # STEP 3: 重算各列保证金（价格变了保证金也变）
    for col in range(N):
        if position[col] != 0:
            new_margin = abs(position[col]) * close[t, col] * mult[col] * margin_rate[col]
            cash -= (new_margin - margin_locked[col])
            margin_locked[col] = new_margin
        else:
            cash += margin_locked[col]
            margin_locked[col] = 0.0

    # STEP 4: 强平
    if cash + sum(margin_locked) <= 0:
        for col in range(N):
            if position[col] != 0:
                do_close(col, price=close[t, col], side=LIQUIDATED)
                liquidated[col] = True

    # STEP 5: 写快照
    out_cash[t] = cash
    out_position[t, :] = position
    out_margin_locked[t, :] = margin_locked
    prev_close[:] = close[t, :]
```

### 5.3 try_open（内联）

```python
def try_open(col, signed_size, price):
    adj_price  = price * (1 + sign(signed_size) * slippage[col])
    notional   = abs(signed_size) * adj_price * mult[col]
    req_margin = notional * margin_rate[col]
    req_fee    = notional * fees[col] + abs(signed_size) * fixed_fees[col]

    if cash >= req_margin + req_fee:
        cash -= (req_margin + req_fee)
        margin_locked[col] += req_margin
        position[col]      += signed_size
        avg_price[col]      = adj_price
        emit_order(side=OPEN_LONG/OPEN_SHORT, ...)
    # else: 拒单, 不留 record
```

### 5.4 do_close（内联）

```python
def do_close(col, price, side=normal):
    adj_price = price * (1 - sign(position[col]) * slippage[col])
    size_abs  = abs(position[col])
    notional  = size_abs * adj_price * mult[col]
    fee_paid  = notional * fees[col] + size_abs * fixed_fees[col]
    realized  = position[col] * (adj_price - avg_price[col]) * mult[col]

    cash               += margin_locked[col]   # 释放保证金
    cash               += realized              # 实现盈亏
    cash               -= fee_paid              # 扣手续费
    margin_locked[col]  = 0.0
    emit_order(side=CLOSE_LONG/CLOSE_SHORT/LIQUIDATED, pnl=realized - fee_paid, ...)
    position[col]   = 0.0
    avg_price[col]  = 0.0
```

### 5.5 边界处理

| 情况 | 处理 |
|---|---|
| `close[t]` 为 NaN | 跳过当根 K 线全部操作，沿用 t-1 快照 |
| 反手第二腿资金不足 | 第一腿（close）已成功；状态合法，记录显示半反手 |
| 全列已强平 | simulator 继续跑（保证 equity 时序长度 = T） |
| `init_cash <= 0` | 包装层抛 `ValueError` |
| `mult / margin_rate <= 0` | 包装层抛 `ValueError` |

---

## 6. FuturesPortfolio 包装类

### 6.1 顶层入口

```python
def from_signals(
    close: pd.DataFrame,
    *,
    long_entries: pd.DataFrame | None = None,
    long_exits: pd.DataFrame | None = None,
    short_entries: pd.DataFrame | None = None,
    short_exits: pd.DataFrame | None = None,
    specs: list[FuturesSpec],
    size: float | np.ndarray | pd.DataFrame = 1.0,
    init_cash: float = 100_000.0,
    freq: str | pd.Timedelta | None = None,
    bars_per_year: float | None = None,
    trading_days_per_year: int = 252,
) -> FuturesPortfolio
```

**`freq` 支持的 pandas 频率字符串**（不止日线）：

| freq | 含义 |
|---|---|
| `"1D"` | 日线 |
| `"4H"` | 4 小时 |
| `"1H"` | 1 小时 |
| `"15min"` / `"15T"` | 15 分钟 |
| `"5min"` / `"5T"` | 5 分钟 |
| `"1min"` / `"1T"` | 1 分钟 |

`freq=None`（默认）会从 `close.index` 推断（用 `pd.infer_freq` 或 index 差值）。

### 6.1.1 年化系数决策树（关键）⭐

按以下优先级确定 `bars_per_year`（用于 Sharpe / Sortino / 年化收益）：

```
1. 用户传入 bars_per_year=N  → 直接用 N           (最高优先级，完全覆盖)
2. 否则按 "交易时段自动推断":
       median_bars_per_day = close.index 按日分组的 bar 数中位数
       bars_per_year = median_bars_per_day × trading_days_per_year
3. 如果 index 不是 DatetimeIndex（无法分组）→ 退化为 freq 日历法 (365 × bars_per_day_by_freq)
```

**默认按"交易时段"推断**（不是日历时段），符合中国期货市场实际：

| 数据 | 自动推断 |
|---|---|
| 日线 | 1 × 252 = **252** |
| 国内日盘 1H (4h/天) | 4 × 252 = **1008** |
| 国内带夜盘 1H (~9.5h/天) | 10 × 252 = **2520** |
| 国内日盘 15min | 16 × 252 = **4032** |
| 国内日盘 5min | 48 × 252 = **12096** |
| 国内日盘 1min | 240 × 252 = **60480** |

**手动覆盖** `bars_per_year`（精确控制 / 跨市场场景）：

```python
# 例：美股 4h（每天 6.5 小时 ÷ 4 ≈ 1.625 个 4h bar × 252）
pf = vbtf.from_signals(..., freq="4H", bars_per_year=410)

# 例：A 股 + 日盘期货混合（用户自己定）
pf = vbtf.from_signals(..., bars_per_year=4032)
```

**`trading_days_per_year`** 用于自动推断时的年化天数：默认 252（A 股/国内期货）。海外股票可传 252，加密 24/7 可传 365。

### 6.1.2 算法实现

```python
# utils.py
def infer_bars_per_year(
    index: pd.DatetimeIndex,
    trading_days_per_year: int = 252,
) -> float:
    """按 index 实际 bar/天 中位数 × 交易日数 推算."""
    df = pd.DataFrame({"date": index.date}, index=index)
    bars_per_day = df.groupby("date").size()
    median_bpd = float(bars_per_day.median())
    return median_bpd * trading_days_per_year
```

用**中位数**抗节假日半日、夜盘缺失、临时停牌噪声。

### 6.2 FuturesPortfolio

```python
@dataclass(frozen=True)
class FuturesPortfolio:
    close:                  pd.DataFrame
    specs:                  tuple[FuturesSpec, ...]
    init_cash:              float
    freq:                   str | pd.Timedelta | None
    bars_per_year:          float                      # 最终生效值 (用户 / 自动推断)
    trading_days_per_year:  int
    _order_records:         np.ndarray
    _cash:                  np.ndarray
    _position:              np.ndarray
    _margin_locked:         np.ndarray
```

### 6.3 派生属性（全部 cached_property）

| 属性 | 类型 |
|---|---|
| `cash` | `pd.Series` |
| `position` | `pd.DataFrame` |
| `margin_locked` | `pd.DataFrame` |
| `equity` | `pd.Series` |
| `returns` | `pd.Series` |
| `drawdown` | `pd.Series` |
| `orders` | `pd.DataFrame` |
| `trades` | `pd.DataFrame` |

### 6.4 trades 配对算法

```
对每列 col:
    open_rec = None
    for r in orders[col] 按时间序:
        if r.side in (OPEN_LONG, OPEN_SHORT):
            open_rec = r
        elif r.side in (CLOSE_LONG, CLOSE_SHORT, LIQUIDATED):
            yield Trade(
                entry_time=open_rec.idx, entry_price=open_rec.price,
                exit_time=r.idx, exit_price=r.price,
                size=open_rec.size,                # 带符号
                pnl=r.pnl,                         # do_close 时算好
                fees=open_rec.fees + r.fees,
                duration_bars=r.idx - open_rec.idx,
                is_liquidated=(r.side == LIQUIDATED),
            )
            open_rec = None
```

反手会产生 [..., OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT, ...]——
配对算法天然正确（CLOSE 紧跟 OPEN）。

### 6.5 stats() 字段

| 字段 | 来源 |
|---|---|
| Start / End / Period | close.index |
| Init Cash | self.init_cash |
| Final Equity | equity.iloc[-1] |
| Total Return [%] | |
| Annualized Return [%] | vbt.returns_accessor（按 `bars_per_year` 覆盖；缺省按 `freq` 日历法） |
| Sharpe Ratio | 同上 |
| Sortino Ratio | 同上 |
| Max Drawdown [%] | drawdown.min() |
| Total Trades | len(trades) |
| Win Rate [%] | |
| Profit Factor | |
| Avg Trade PnL | |
| **Avg Win** | trades.loc[pnl>0, "pnl"].mean() |
| **Avg Loss** | trades.loc[pnl<0, "pnl"].mean() |
| **Win/Loss Ratio** | Avg Win / abs(Avg Loss) |
| **Max Position** | max(abs(position)).max() |
| Total Fees | orders.fees.sum() |
| Liquidations | (trades.is_liquidated).sum() |
| Bars Per Year | self.bars_per_year（实际生效值，自动推断 or 用户传入） |
| Trading Days Per Year | self.trading_days_per_year |

### 6.6 plot()

- 双面板 plotly
- 上面板：所有品种 close 线（不限数量）+ 开/平/反手/强平标记
- 下面板：equity 曲线 + drawdown 红色填充

### 6.7 调试辅助

- `to_vbt_orders()` — 转 vbt order_dt 格式，仅供对比
- 不提供 `to_vbt_portfolio()`（用户已确认不需要）

---

## 7. 错误处理

### 7.1 包装层校验（抛 `ValueError`）

| 校验 | 错误信息样例 |
|---|---|
| len(specs) != close.shape[1] | `"len(specs)=3 不匹配 close 的列数 5"` |
| margin_rate ≤ 0 | `"FuturesSpec(RB).margin_rate 必须 > 0"` |
| mult ≤ 0 | 同上 |
| init_cash ≤ 0 | `"init_cash 必须 > 0, 收到 -100"` |
| 信号 shape 不匹配 | `"long_entries 形状 (1000, 3) 不匹配 close (1000, 5)"` |
| 信号非 bool | `"long_entries 必须 bool 类型, 收到 int64"` |
| size 含 ≤0 | `"size 含非正数 -2.0; 方向由信号决定, size 必须 > 0"` |
| close 含负值 | `"close[5, 2] = -3.0, 期货价格必须 > 0"` |
| flat_conflict 非法 | `"flat_conflict 必须是 'long'|'short'|'skip', 收到 'random'"` |
| `freq` 无法解析 | `"freq='banana' 不能解析为 pd.Timedelta"` |
| `bars_per_year` ≤ 0 | `"bars_per_year 必须 > 0, 收到 -10"` |
| `trading_days_per_year` ≤ 0 | `"trading_days_per_year 必须 > 0, 收到 0"` |
| 自动推断时 index 非 DatetimeIndex 且未传 freq | `"无法推断 bars_per_year: index 不是 DatetimeIndex 且未传 freq; 请显式传 bars_per_year"` |

所有校验集中在 `utils.py:_validate_inputs()`，前置失败。

### 7.2 Simulator 内部不抛错

- 拒单 → 不写 record
- 强平 → 写 `side=LIQUIDATED` 的 close record
- NaN close → 跳过 + 沿用 t-1 快照
- 仅用 `assert` 守内部不变量（如 `position==0 ⇔ margin_locked==0`）

### 7.3 FuturesPortfolio 层不应失败

- `stats()` 在 trades 为空时返回 NaN，不抛

---

## 8. 测试策略

遵循全局规则：**100% 测试覆盖率、pytest、AAA 模式、TDD**。

### 8.1 测试金字塔

| 层 | 文件 | 用例数（≥） | 说明 |
|---|---|---|---|
| 1 simulator 单元 | test_simulator.py | 11 | 直接调 njit, 数字精确比对 |
| 2 边界 / 校验 | test_margin.py | 6 | 保证金动态变化 + 校验失败 |
| 3 强平 | test_liquidation.py | 4 | 单列 / 多列 / 强平后状态 |
| 4 信号规则 | test_signals.py | 8 | §5.1 全场景对照表逐行验证（两阶段评估） |
| 5 端到端 | test_portfolio.py | 9 | from_signals 全流程 + stats / plot |
| 6 派生属性 | test_derived.py | 7 | cash/position/equity/returns/drawdown/orders/trades 不变量 |
| 7 校验失败 | test_validation.py | 13 | §7.1 每条错误信息都触发一次 |
| 8 频率与年化 | test_freq.py | 12 | 自动推断 + 用户覆盖 + 边界 |
| 9 vs backtrader | test_vs_backtrader.py | 骨架 | marker=slow, 允许小误差 |

### 8.2 关键测试用例（举例，不完全列表）

```
test_open_long_consumes_margin
test_close_long_releases_margin_and_books_pnl
test_mark_to_market_increases_cash_when_price_rises
test_dynamic_margin_recompute_on_price_change
test_reversal_long_to_short_emits_two_records
test_reject_when_cash_insufficient
test_liquidation_when_equity_zero_or_below
test_no_new_orders_after_liquidation
test_trades_pairing_with_reversal
test_stats_returns_expected_fields
test_long_exit_then_long_entry_same_bar_emits_two_records  # §5.1 同向再入
test_flat_conflict_skip_when_both_entries_true             # 默认行为
test_flat_conflict_long_when_configured                    # 替代行为
test_flat_conflict_short_when_configured
test_nan_close_skips_bar
test_plot_returns_plotly_figure_with_expected_traces
test_freq_1D_default_annualization                          # freq 处理
test_freq_15min_calendar_annualization
test_freq_1H_calendar_annualization
test_bars_per_year_override_takes_precedence                # bars_per_year 覆盖
test_freq_invalid_string_raises_validation_error
test_bars_per_year_non_positive_raises
test_stats_bars_per_year_field_shows_actual_value           # stats 字段呈现
test_infer_bars_per_year_daily_returns_252                  # 自动推断
test_infer_bars_per_year_1h_day_session_returns_1008        # 国内日盘 1H
test_infer_bars_per_year_15min_returns_4032
test_infer_bars_per_year_uses_median_robust_to_half_day     # 节假日半日
test_trading_days_per_year_custom_value                     # 365 for crypto
test_non_datetime_index_without_freq_raises                 # 边界
```

### 8.3 fixtures（conftest.py）

- `simple_spec` — mult=10, margin=10%, 零费率（数学最干净）
- `realistic_rb_spec` — mult=10, margin=10%, fees=2e-4, slippage=1e-4
- `synthetic_close_1col(T=100)` / `synthetic_close_3col(T=500)`
- `deterministic_seed=42` — 所有随机源固定种子，保证可复现

### 8.4 100% 覆盖率策略

为达到 100% 行覆盖率：

| 难点 | 应对 |
|---|---|
| **njit 函数覆盖率** | numba 编译过的函数 `coverage.py` 默认看不到。在测试环境用环境变量 `NUMBA_DISABLE_JIT=1` 跑覆盖率收集，等同跑纯 Python；性能测试单独跑（NUMBA 开启） |
| **`plot()` 内部 plotly 渲染** | 不测 plotly 内部细节，只断言返回 Figure 且 traces 数 == 预期 |
| **不可达分支** | 用 `# pragma: no cover` 显式标记（如 `if False: ...` 守护或 `__repr__` 的兜底） |
| **assert 守 invariant** | 不能进入的分支用 `# pragma: no cover` 标记 |
| **异常分支** | `test_validation.py` 逐条触发每个 ValueError |

`pytest.ini` 配置：
```ini
[pytest]
addopts =
    --cov=src/vbt_futures
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=100
    --cov-branch
markers =
    slow: 跨工具对比测试 (vs backtrader)
```

CI 跑：`NUMBA_DISABLE_JIT=1 pytest -m "not slow"`，本地额外跑 `pytest -m slow` 验证。

### 8.5 性能验证（非 CI）

`benchmarks/bench_simulator.py`：1000 bars × 5 cols × 100 iter，目标单次 < 5ms（**NUMBA_DISABLE_JIT 关闭**）。

---

## 9. 命名约定

| 类别 | 约定 | 例 |
|---|---|---|
| 模块 / 文件 | snake_case | `simulator.py` |
| 类 | PascalCase | `FuturesSpec` |
| 函数 | snake_case | `from_signals` |
| njit 函数 | snake_case + `_nb` | `simulate_futures_nb` |
| 常量 / 枚举 | UPPER_SNAKE | `OPEN_LONG` |
| record dtype | snake_case + `_dt` | `futures_order_dt` |
| 测试 | `test_<expected_behavior>` | `test_reversal_long_to_short_emits_two_records` |
| 私有 | 单下划线 | `_validate_inputs` |

---

## 10. 依赖

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "numba>=0.58",
    "vectorbt>=0.25",
    "plotly>=5.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "ruff>=0.1", "mypy>=1.5"]
data = ["akshare"]
backtest_compare = ["backtrader>=1.9"]
```

安装：
```bash
"C:/Users/97554/Desktop/backtrader/.venv/Scripts/python.exe" -m pip install -e vbt-futures[dev]
```

---

## 11. 公共 API

```python
import vbt_futures as vbtf

# ── 场景 1: 日线 (默认 252 bars/year, 无需任何配置) ────────────────
pf = vbtf.from_signals(
    close=df_close,                        # DatetimeIndex 日级
    long_entries=df_long_ent,
    long_exits=df_long_ex,
    short_entries=df_short_ent,
    short_exits=df_short_ex,
    specs=[
        vbtf.FuturesSpec("RB", mult=10, margin_rate=0.10, fees=2e-4),
        vbtf.FuturesSpec("HC", mult=10, margin_rate=0.10, fees=2e-4),
        vbtf.FuturesSpec("I",  mult=100, margin_rate=0.12, fees=2e-4),
    ],
    size=1.0,
    init_cash=200_000.0,
    # freq/bars_per_year 都不传 → 自动推断 = 1 bar/day × 252 = 252
)

# ── 场景 2: 国内日盘 1 小时 (自动 → 4 × 252 = 1008) ────────────────
pf_1h = vbtf.from_signals(
    close=df_close_1h,                     # 每天 9-15 点 4 根 bar
    long_entries=df_le, long_exits=df_lx,
    short_entries=df_se, short_exits=df_sx,
    specs=[vbtf.FuturesSpec("RB", mult=10, margin_rate=0.10, fees=2e-4)],
    size=1.0,
    init_cash=200_000.0,
    # 自动推断 bars_per_year = 1008
)

# ── 场景 3: 美股 4H (手动覆盖 trading_days 不变, 自定 bars_per_year) ──
pf_us = vbtf.from_signals(
    close=df_close_us_4h,
    ...,
    bars_per_year=410,                     # 显式覆盖
)

# ── 场景 4: 加密 24/7 (改 trading_days) ────────────────────────────
pf_crypto = vbtf.from_signals(
    close=df_close_crypto_1h,
    ...,
    trading_days_per_year=365,            # 自动变 24 × 365 = 8760
)

pf.cash; pf.position; pf.equity; pf.returns; pf.drawdown
pf.orders; pf.trades
pf.stats(); pf.plot(); pf.to_vbt_orders()
```

---

## 12. 已知限制（README 要说明）

1. **跳空盈亏 ✅ 正确捕捉**（公式 `pnl = position × (close[t] − close[t-1]) × mult` 天然吃下所有 close-to-close 价差，含隔夜/周末跳空），**但有 3 处近似**：
   - 盯市基准用 **close** 而非交易所**结算价（settlement price）**——多数策略影响 < 0.1%
   - **强平价**用 K 线末 close；若跳空开盘瞬间已破净，实盘会在开盘附近强平、价格更差，本框架会"等到 K 线末"才强平 → 强平价偏乐观
   - **不读 OHLC 高低点**——所有信号、止损、止盈触发都基于 close；想用 high/low 盘中触发需自己用其他工具预处理出信号
2. 不支持移仓换月（用户拼连续合约）
3. **年化指标默认按"交易时段"自动推断**（按 `close.index` 实际 bar/天中位数 × 252）。
   - 国内日盘 1H 数据 → 1008 个 bar/年
   - 带夜盘 1H 数据 → 2520 个 bar/年
   - 日线 → 252
   - 详见 §6.1.1 决策树。
   - 可用 `bars_per_year` / `trading_days_per_year` 覆盖（如美股 252、加密 365）。
4. 不做 SPAN 组合保证金
5. 不支持多空锁仓（单边持仓）

---

## 13. 验收标准

- [ ] **所有测试通过且行覆盖率 + 分支覆盖率均 = 100%**
      （`pytest --cov-fail-under=100 --cov-branch`，CI 跑 `NUMBA_DISABLE_JIT=1`）
- [ ] `examples/demo_synthetic.py` 能跑出 stats() + 生成 plot html
- [ ] 单次 simulate_futures_nb（1000 bars × 5 cols，NUMBA 开启）< 5ms
- [ ] README 包含 API 文档 + 与 vbt / bt 的差异说明
- [ ] 所有文件 ≤ 800 行
- [ ] 所有未覆盖到的不可达分支用 `# pragma: no cover` 显式标注，code review 时确认合理
