# vbt-futures 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 vectorbt 的 numba 框架之上，新建独立模块 `vbt-futures`，实现 backtrader 风格的期货回测（保证金、合约乘数、逐日盯市、多空反手、强平），对外暴露 `from_signals` 风格 API。

**Architecture:** 自写 @njit `simulate_futures_nb` 处理期货语义；复用 vbt 的 `returns_accessor` 做指标；输出 `FuturesPortfolio` Python 包装类。

**Tech Stack:** Python 3.10+、numpy、pandas、numba、vectorbt、plotly、pytest。

---

## Context（为什么做这件事）

- **backtrader** 原生支持期货但 2019 年后停更、性能差。
- **vectorbt** 高性能但内核是股票模型（`vectorbt/portfolio/nb.py:148` 直接按 `size * price` 全额扣现金，不认 margin/mult），无法直接用于期货。
- **vbt-futures** 把两者的优点拼起来：vbt 的 numba 引擎 + backtrader 的期货语义。
- 详细设计已经在 spec 里定稿，本计划只负责把 spec 翻译成可执行的 TDD 步骤。

**Spec 路径**：`C:\Users\97554\Desktop\backtrader\vbt-futures\docs\superpowers\specs\2026-06-11-vbt-futures-design.md`

---

## 关键约束（计划执行时必须遵守）

| 项 | 要求 |
|---|---|
| 覆盖率 | 行 + 分支 **100%**（`pytest --cov-fail-under=100 --cov-branch`） |
| 测试模式 | 严格 TDD（红 → 绿 → 重构 → 提交） |
| numba 与覆盖率冲突 | 单元测试运行时 `NUMBA_DISABLE_JIT=1`；性能 benchmark 单独跑（NUMBA 开） |
| 文件大小 | 每个文件 ≤ 800 行 |
| Python 风格 | PEP 8、type annotations、`@dataclass(frozen=True)`、black/ruff |
| 工作目录 | `C:\Users\97554\Desktop\backtrader\vbt-futures\` |
| Python 解释器 | `C:/Users/97554/Desktop/backtrader/.venv/Scripts/python.exe`（已装 numpy/pandas，未装 numba/vectorbt） |

---

## 文件结构（参考 spec §3.1）

```
vbt-futures/
├── pyproject.toml
├── pytest.ini
├── README.md
├── .gitignore
├── src/vbt_futures/
│   ├── __init__.py        # 公开 API: from_signals, FuturesSpec, FuturesPortfolio
│   ├── enums.py           # OPEN_LONG/CLOSE_LONG/.../LIQUIDATED + FlatConflict 常量
│   ├── records.py         # futures_order_dt
│   ├── spec.py            # FuturesSpec dataclass
│   ├── utils.py           # _validate_inputs, infer_bars_per_year, broadcast helpers
│   ├── simulator.py       # @njit simulate_futures_nb (核心 ~250 行)
│   └── portfolio.py       # FuturesPortfolio + 派生属性 + stats + plot
├── tests/
│   ├── conftest.py        # simple_spec / realistic_rb_spec / synthetic_close fixtures
│   ├── test_simulator.py
│   ├── test_margin.py
│   ├── test_liquidation.py
│   ├── test_signals.py
│   ├── test_portfolio.py
│   ├── test_derived.py
│   ├── test_validation.py
│   ├── test_freq.py
│   └── test_vs_backtrader.py   # marker=slow
├── examples/demo_synthetic.py
├── benchmarks/bench_simulator.py
└── docs/superpowers/specs/2026-06-11-vbt-futures-design.md  # 已存在
```

---

## 复用清单（避免重写）

| 来源 | 用法 | 引入方式 |
|---|---|---|
| `vectorbt.portfolio.enums.SizeType / Direction` | 仅供参考与互转 | `from vectorbt.portfolio.enums import ...` |
| `vectorbt.returns_accessor`（`.vbt.returns`） | Sharpe / Sortino / Annualized | `series.vbt.returns(freq=...)` |
| `numba.njit` | 核心 simulator | `from numba import njit` |
| `numpy` record dtype | 自定义 `futures_order_dt`（不复用 vbt 的 order_dt，缺字段） | `np.dtype([...], align=True)` |
| **不复用**：vbt 的 `buy_nb / sell_nb / process_order_nb` | 内核假设是股票（见 `vectorbt/portfolio/nb.py:148`），不兼容 |

---

# 实施任务（按依赖顺序）

## Phase 1：项目脚手架

### Task 1：初始化项目目录 + git

**Files:**
- Create: `vbt-futures/.gitignore`
- Create: `vbt-futures/pyproject.toml`
- Create: `vbt-futures/pytest.ini`
- Create: `vbt-futures/README.md`（占位）
- Create: `vbt-futures/src/vbt_futures/__init__.py`（占位）

- [ ] **Step 1.1:** `cd C:/Users/97554/Desktop/backtrader/vbt-futures && git init && git checkout -b main`
- [ ] **Step 1.2:** 写 `.gitignore`（Python + venv + .coverage + __pycache__/ + .mypy_cache/ + .ruff_cache/ + numba cache）
- [ ] **Step 1.3:** 写 `pyproject.toml`（按 spec §10）
- [ ] **Step 1.4:** 写 `pytest.ini`（按 spec §8.4）：
      ```ini
      [pytest]
      addopts = --cov=src/vbt_futures --cov-report=term-missing --cov-report=html --cov-fail-under=100 --cov-branch
      markers =
          slow: 跨工具对比测试 (vs backtrader)
      env =
          NUMBA_DISABLE_JIT=1
      ```
- [ ] **Step 1.5:** 创建空目录骨架：`src/vbt_futures/`、`tests/`、`examples/`、`benchmarks/`
- [ ] **Step 1.6:** `git add . && git commit -m "chore: project scaffold"`

### Task 2：安装依赖

- [ ] **Step 2.1:** `"C:/Users/97554/Desktop/backtrader/.venv/Scripts/python.exe" -m pip install -e "vbt-futures[dev]"`
- [ ] **Step 2.2:** 跑 `pytest --collect-only` 确认空环境能起来
- [ ] **Step 2.3:** Commit lock 文件（如有）

---

## Phase 2：基础模块（无 numba 依赖）

### Task 3：`enums.py` —— side 常量 + FlatConflict 映射

**Files:**
- Create: `src/vbt_futures/enums.py`
- Test: `tests/test_enums.py`

- [ ] **Step 3.1:** 写测试 `test_enums.py`：断言 OPEN_LONG=0, CLOSE_LONG=1, OPEN_SHORT=2, CLOSE_SHORT=3, LIQUIDATED=4；`FLAT_CONFLICT_CODE["skip"]==2` 等
- [ ] **Step 3.2:** 运行测试 → FAIL
- [ ] **Step 3.3:** 实现 `enums.py`：
      ```python
      from typing import Final
      OPEN_LONG: Final[int]   = 0
      CLOSE_LONG: Final[int]  = 1
      OPEN_SHORT: Final[int]  = 2
      CLOSE_SHORT: Final[int] = 3
      LIQUIDATED: Final[int]  = 4

      FLAT_CONFLICT_CODE: Final[dict[str, int]] = {"long": 0, "short": 1, "skip": 2}
      ```
- [ ] **Step 3.4:** 运行测试 → PASS
- [ ] **Step 3.5:** `git commit -m "feat(enums): side codes and flat-conflict mapping"`

### Task 4：`records.py` —— futures_order_dt

**Files:**
- Create: `src/vbt_futures/records.py`
- Test: `tests/test_records.py`

- [ ] **Step 4.1:** 写测试：构造 1 条 record，断言字段名/dtype/可读写、对齐
- [ ] **Step 4.2:** FAIL
- [ ] **Step 4.3:** 实现按 spec §4.4 的 9 字段 dtype，align=True
- [ ] **Step 4.4:** PASS
- [ ] **Step 4.5:** `git commit -m "feat(records): futures_order_dt"`

### Task 5：`spec.py` —— FuturesSpec dataclass

**Files:**
- Create: `src/vbt_futures/spec.py`
- Test: `tests/test_spec.py`

- [ ] **Step 5.1:** 写测试：构造、frozen、默认值、`flat_conflict` 取值范围
- [ ] **Step 5.2:** FAIL
- [ ] **Step 5.3:** 实现 `@dataclass(frozen=True)`，字段按 spec §4.1
- [ ] **Step 5.4:** PASS
- [ ] **Step 5.5:** `git commit -m "feat(spec): FuturesSpec dataclass"`

### Task 6：`utils.py::infer_bars_per_year`

**Files:**
- Create: `src/vbt_futures/utils.py`
- Test: `tests/test_freq.py`（部分）

- [ ] **Step 6.1:** 写 4 条测试（日线=252 / 国内日盘 1H=1008 / 15min=4032 / 节假日半日中位数鲁棒）
- [ ] **Step 6.2:** FAIL
- [ ] **Step 6.3:** 实现按 spec §6.1.2
- [ ] **Step 6.4:** PASS
- [ ] **Step 6.5:** `git commit -m "feat(utils): infer_bars_per_year"`

### Task 7：`utils.py::_validate_inputs`

**Files:**
- Modify: `src/vbt_futures/utils.py`
- Test: `tests/test_validation.py`

- [ ] **Step 7.1:** 写 13 条 ValueError 触发测试（按 spec §7.1 表）
- [ ] **Step 7.2:** FAIL
- [ ] **Step 7.3:** 实现 `_validate_inputs(close, long_entries, long_exits, short_entries, short_exits, specs, size, init_cash, freq, bars_per_year, trading_days_per_year) -> None`
- [ ] **Step 7.4:** PASS
- [ ] **Step 7.5:** `git commit -m "feat(utils): validate inputs"`

---

## Phase 3：核心 simulator（按功能切片增量做）

### Task 8：`simulator.py` 骨架（无信号、纯初始化）

**Files:**
- Create: `src/vbt_futures/simulator.py`
- Test: `tests/test_simulator.py`

- [ ] **Step 8.1:** 写测试 `test_no_signals_returns_init_state`：close 全 100、所有信号全 False、应得 `cash[T-1]==init_cash`、`position` 全 0、`orders` 空、`margin_locked` 全 0
- [ ] **Step 8.2:** FAIL
- [ ] **Step 8.3:** 实现 `simulate_futures_nb` 签名（按 spec §4.2/4.3）+ 主循环 STEP 1/3/4/5 框架（STEP 2 留空），开 `@njit(cache=True)`
- [ ] **Step 8.4:** PASS
- [ ] **Step 8.5:** Commit

### Task 9：开多（try_open 内联到 simulator）

**Files:**
- Modify: `src/vbt_futures/simulator.py`
- Test: `tests/test_simulator.py`

- [ ] **Step 9.1:** 写 `test_open_long_consumes_margin`（spec §8.2 第 1 条原文示例）：close=[100,100], long_entries=[True,False]，断言 t=0 cash=9900, margin_locked=100
- [ ] **Step 9.2:** FAIL
- [ ] **Step 9.3:** STEP 2 实现 `pass1` + `pass2`，flat → long_entry → try_open 分支（spec §5.3 公式）
- [ ] **Step 9.4:** PASS
- [ ] **Step 9.5:** Commit

### Task 10：盯市

- [ ] **Step 10.1:** 写 `test_mark_to_market_increases_cash_when_price_rises`（spec §8 示例）
- [ ] **Step 10.2:** FAIL
- [ ] **Step 10.3:** STEP 1 启用 mark-to-market 公式
- [ ] **Step 10.4:** PASS + commit

### Task 11：动态保证金重算

- [ ] **Step 11.1:** 写 `test_dynamic_margin_recompute_on_price_change`：价格涨 margin_locked 同步变
- [ ] **Step 11.2:** FAIL
- [ ] **Step 11.3:** STEP 3 启用差额扣现金
- [ ] **Step 11.4:** PASS + commit

### Task 12：平多 + 实现盈亏（do_close 内联）

- [ ] **Step 12.1:** 写 `test_close_long_releases_margin_and_books_pnl`
- [ ] **Step 12.2:** FAIL
- [ ] **Step 12.3:** PASS 1 加 long_exit → CLOSE 分支；实现 do_close 公式（spec §5.4）
- [ ] **Step 12.4:** PASS + commit

### Task 13：开空 + 平空

- [ ] **Step 13.1:** 写 `test_open_short_consumes_margin` + `test_close_short_releases_margin_and_books_pnl`
- [ ] **Step 13.2:** FAIL
- [ ] **Step 13.3:** flat → short_entry → try_open(-size)；持空 → short_exit → do_close
- [ ] **Step 13.4:** PASS + commit

### Task 14：反手（PASS 1 的 REVERSE_TO_*）

**Files:**
- Test: `tests/test_signals.py`

- [ ] **Step 14.1:** 写 `test_reversal_long_to_short_emits_two_records` 和 `test_reversal_short_to_long_emits_two_records`：断言 orders 中正好出现 2 条（先 CLOSE 后 OPEN，同一 idx）
- [ ] **Step 14.2:** FAIL
- [ ] **Step 14.3:** PASS 1 实现 short_entry（持多时）→ REVERSE_TO_SHORT 分支：先 do_close 再 try_open(-size)，`continue` 跳过 PASS 2
- [ ] **Step 14.4:** PASS + commit

### Task 15：同向止盈再入（long_exit + long_entry 同 bar）

- [ ] **Step 15.1:** 写 `test_long_exit_then_long_entry_same_bar_emits_two_records`：起始持多，触发同 bar 两信号；断言 t 出现 CLOSE_LONG 紧跟 OPEN_LONG
- [ ] **Step 15.2:** FAIL
- [ ] **Step 15.3:** 验证 PASS 1 CLOSE → 落 PASS 2 → OPEN 逻辑正确（应已通过 Task 14 间接验证；不通过则修 PASS 2 入口判断 `position[col]==0`）
- [ ] **Step 15.4:** PASS + commit

### Task 16：flat_conflict 三档（long / short / skip）

- [ ] **Step 16.1:** 写 3 条测试：`test_flat_conflict_skip_when_both_entries_true` / `_long` / `_short`
- [ ] **Step 16.2:** FAIL
- [ ] **Step 16.3:** PASS 2 实现 long_entry & short_entry 同 True 时 switch by flat_conflict_code
- [ ] **Step 16.4:** PASS + commit

### Task 17：拒单（保证金不足）

**Files:**
- Test: `tests/test_margin.py`

- [ ] **Step 17.1:** 写 `test_reject_when_cash_insufficient`：init_cash=50, 开仓需要保证金 100, 断言 orders 空、position 仍 0
- [ ] **Step 17.2:** FAIL
- [ ] **Step 17.3:** try_open 加资金检查 `if cash >= req_margin + req_fee else 跳过不留 record`
- [ ] **Step 17.4:** PASS + commit

### Task 18：强平（总权益 ≤ 0）

**Files:**
- Test: `tests/test_liquidation.py`

- [ ] **Step 18.1:** 写 4 条测试：`test_liquidate_single_column` / `test_liquidate_multiple_columns_same_bar` / `test_no_new_orders_after_liquidation` / `test_liquidation_preserves_equity_zero`
- [ ] **Step 18.2:** FAIL
- [ ] **Step 18.3:** STEP 4 实现：检查 total_equity ≤ 0 → 对所有 position!=0 的列 do_close(side=LIQUIDATED) + 置 liquidated[col]=True
- [ ] **Step 18.4:** STEP 2 入口加 `if liquidated[col]: continue`
- [ ] **Step 18.5:** PASS + commit

### Task 19：NaN close 跳过

- [ ] **Step 19.1:** 写 `test_nan_close_skips_bar`
- [ ] **Step 19.2:** FAIL
- [ ] **Step 19.3:** 主循环开头 `for col: if np.isnan(close[t,col]): 跳过 col 当根全部操作`
- [ ] **Step 19.4:** PASS + commit

### Task 20：滑点 + 百分比 / 固定手续费

- [ ] **Step 20.1:** 写 3 条测试：滑点对开多平多价格的方向、fees=2e-4 + fixed_fees=3 的混合扣费、零费率 sanity
- [ ] **Step 20.2:** FAIL
- [ ] **Step 20.3:** try_open / do_close 内 `adj_price = price * (1 ± sign(signed_size) * slippage[col])`；`fee = notional * fees + size_abs * fixed_fees`
- [ ] **Step 20.4:** PASS + commit

---

## Phase 4：FuturesPortfolio 包装

### Task 21：portfolio.py 骨架 + 基础属性

**Files:**
- Create: `src/vbt_futures/portfolio.py`
- Test: `tests/test_derived.py`

- [ ] **Step 21.1:** 写 5 条测试：cash / position / margin_locked / equity / returns / drawdown 各属性形状、`equity == cash + margin_locked.sum(axis=1)` 不变量
- [ ] **Step 21.2:** FAIL
- [ ] **Step 21.3:** 实现 `@dataclass(frozen=True) FuturesPortfolio`（spec §6.2 字段）+ cached_property 6 个
- [ ] **Step 21.4:** PASS + commit

### Task 22：orders / trades DataFrames

- [ ] **Step 22.1:** 写 `test_orders_dataframe_columns_and_dtypes` + `test_trades_pairing` + `test_trades_pairing_with_reversal`
- [ ] **Step 22.2:** FAIL
- [ ] **Step 22.3:** 实现 `orders` cached_property（_order_records → DataFrame 加 symbol/datetime 列）+ `trades` cached_property（spec §6.4 配对算法）
- [ ] **Step 22.4:** PASS + commit

### Task 23：stats()

**Files:**
- Test: `tests/test_portfolio.py`

- [ ] **Step 23.1:** 写 `test_stats_returns_expected_fields`（断言 18 个字段都存在）
- [ ] **Step 23.2:** FAIL
- [ ] **Step 23.3:** 实现 `stats()` 返回 pd.Series，借用 `self.returns.vbt.returns(freq=self.freq).sharpe_ratio()`（按 bars_per_year）等
- [ ] **Step 23.4:** PASS + commit

### Task 24：plot()

- [ ] **Step 24.1:** 写 `test_plot_returns_plotly_figure_with_expected_traces`
- [ ] **Step 24.2:** FAIL
- [ ] **Step 24.3:** 实现 `plot()` 双面板：上 close 多线 + 标记，下 equity + drawdown
- [ ] **Step 24.4:** PASS + commit

### Task 25：to_vbt_orders()

- [ ] **Step 25.1:** 写 `test_to_vbt_orders_returns_dataframe`
- [ ] **Step 25.2:** FAIL
- [ ] **Step 25.3:** 实现 mapping（OPEN_LONG/CLOSE_SHORT → Buy=0；OPEN_SHORT/CLOSE_LONG → Sell=1）
- [ ] **Step 25.4:** PASS + commit

---

## Phase 5：公开 API & 端到端

### Task 26：`from_signals` + `__init__.py`

**Files:**
- Modify: `src/vbt_futures/__init__.py`
- Test: `tests/test_portfolio.py`

- [ ] **Step 26.1:** 写 `test_from_signals_smoke`（3 个品种 + 简单信号端到端）+ `test_from_signals_propagates_validation_errors`
- [ ] **Step 26.2:** FAIL
- [ ] **Step 26.3:** 实现 `from_signals`：校验 → 拆 specs → 调 `infer_bars_per_year`（或用 user 传值）→ DataFrame → numpy → 调 `simulate_futures_nb` → 包成 `FuturesPortfolio` 返回
- [ ] **Step 26.4:** PASS + commit

### Task 27：freq / bars_per_year 集成测试

**Files:**
- Test: `tests/test_freq.py`（补充剩余 8 条）

- [ ] **Step 27.1:** 写 8 条剩余测试（spec §8.2 freq 部分）
- [ ] **Step 27.2:** FAIL（或 PASS 如逻辑已写完）
- [ ] **Step 27.3:** 修最后的边界
- [ ] **Step 27.4:** PASS + commit

---

## Phase 6：Demo + 收尾

### Task 28：合成数据 demo

**Files:**
- Create: `examples/demo_synthetic.py`

- [ ] **Step 28.1:** 写 demo：3 个品种 × 500 天 × 双均线策略
- [ ] **Step 28.2:** 跑通：`python examples/demo_synthetic.py` → 看到 stats 表 + HTML
- [ ] **Step 28.3:** Commit

### Task 29：README

**Files:**
- Modify: `README.md`

- [ ] **Step 29.1:** 写 quick-start + API + 与 backtrader/vbt 差异（spec §1.3 + §12 摘录）
- [ ] **Step 29.2:** Commit

### Task 30：性能 benchmark

**Files:**
- Create: `benchmarks/bench_simulator.py`

- [ ] **Step 30.1:** 1000 bars × 5 cols × 100 iter，目标 < 5ms/iter
- [ ] **Step 30.2:** **不带 NUMBA_DISABLE_JIT** 跑：`python benchmarks/bench_simulator.py`
- [ ] **Step 30.3:** 若超 5ms：profile 找热点；可加 `parallel=True` 但需重测
- [ ] **Step 30.4:** Commit

### Task 31：100% 覆盖率收尾

- [ ] **Step 31.1:** 跑 `NUMBA_DISABLE_JIT=1 pytest -m "not slow"`，看 term-missing 报告
- [ ] **Step 31.2:** 对每个未覆盖行：要么补测试，要么标 `# pragma: no cover` 并在 PR 描述里说明理由
- [ ] **Step 31.3:** 再跑直到 coverage=100% 通过
- [ ] **Step 31.4:** Commit

### Task 32（可选）：vs backtrader 对比

**Files:**
- Create: `tests/test_vs_backtrader.py`

- [ ] **Step 32.1:** 单品种 + 简单 SMA 交叉策略，在 backtrader 和 vbt-futures 里跑同一组合成数据
- [ ] **Step 32.2:** 断言 final equity 误差 < 1% 即可（不强求 1e-6）
- [ ] **Step 32.3:** 加 `@pytest.mark.slow`
- [ ] **Step 32.4:** Commit

---

## Verification（端到端验证清单）

按以下顺序逐项跑：

1. **依赖装好**
   ```bash
   "C:/Users/97554/Desktop/backtrader/.venv/Scripts/python.exe" -m pip install -e "C:/Users/97554/Desktop/backtrader/vbt-futures[dev]"
   ```

2. **全测试 + 100% 覆盖率**
   ```bash
   cd C:/Users/97554/Desktop/backtrader/vbt-futures
   NUMBA_DISABLE_JIT=1 pytest -m "not slow"
   ```
   期望：所有测试 PASS，coverage = 100%。

3. **Demo 跑通**
   ```bash
   "C:/Users/97554/Desktop/backtrader/.venv/Scripts/python.exe" examples/demo_synthetic.py
   ```
   期望：终端看到 stats 表；`output/demo_synthetic.html` 生成、能在浏览器里看到双面板。

4. **性能基准**
   ```bash
   "C:/Users/97554/Desktop/backtrader/.venv/Scripts/python.exe" benchmarks/bench_simulator.py
   ```
   期望：单次 simulate_futures_nb（1000×5）< 5ms（NUMBA 编译过后）。

5. **可选：与 backtrader 对比**
   ```bash
   pytest -m slow
   ```

---

## 验收闸门

- [ ] `pytest --cov-fail-under=100 --cov-branch` 全绿
- [ ] `demo_synthetic.py` 能产出 stats + HTML
- [ ] simulate_futures_nb 单次 ≤ 5ms（1000×5）
- [ ] 所有 src/ 文件 ≤ 800 行
- [ ] README 包含 quick-start + 已知限制（spec §12）
- [ ] 所有 `# pragma: no cover` 在 review 时确认合理
