# nvidb refactor plan

> ## 0. 执行进度（更新于 2026-08-17）
>
> 工作树干净，所有已完成阶段均已提交，`pytest tests/` 全绿（455 passed，阶段 3 新增
> 42 个 webdata 表征测试）。原始计划正文保留在下方，行号基于扫描时的树，后续阶段请按
> 符号名定位。
>
> ### 已完成
>
> | 阶段 | 提交 | 内容 |
> |---|---|---|
> | 前置 | `2332136` | tokscale 风格 hairline 视觉重构 + 零闪烁差分渲染（`tui_theme.py`） |
> | 前置 | `6e285ef` | NODES 视图默认全展开 + 按终端高度自适应缩放（`_scale_server_blocks`） |
> | 阶段 1 | `a5dde1f` | 零风险删除：~730 行（run.py 调试入口/别名/init 双解析；connection.py 的 nvidbInit、pool 级 execute_command 系列、BaseClient 死方法、双 get_client、swap 死解析；utils.py 损坏查询助手；data_modules.py 的 GPUProcess/GPUInfo + 注释块【使 `import nvidb` 不再经死路径拉起 paramiko+pandas】；sched 的 TransportPool、write_file、purge_alerts、只写不读的 LaunchResult 字段、tick/_terminate/get_meta/purge_jobs/probe 死参数、未使用的 Queue/open_queue API；`build/`、`nvidb.egg-info/`、`nvidb/src/`、example_interactive.py、误导性 print；ruff F401/F841 清扫）。注意：`ServerListInfo.to_dict` 被测试使用，已保留 |
> | 阶段 2 | `70e59b5` | 不可达兜底：connection.py 145 处 `getattr(self,…)` + 5 处 hasattr → 直接属性访问（唯一真懒加载 `_tui_diff_screen` 保留 getattr；测试 fixture 用 `__new__` 绕过 `__init__` 但已设齐属性，无需改）；connect() 重复 except 合并；`x[:11] if len(x)>11` 等 no-op 化简；键盘监听器崩溃改为记日志；dashboard.py import fallback 收敛；monitor.py `PYNVML_AVAILABLE` 删除；scheduler.py `(self.cfg or {})` → `self.cfg`；队列 TUI 配置同步失败改走 error 通道；metrics.py None 守卫删除；dcgm.py WatchFields/UpdateAllFields 共用 except 拆分 |
> | 阶段 3 | （待提交，见下） | 删除 web.py 的 Streamlit UI（~1,320 行），保留的 ~660 行数据层改名为 `nvidb/webdata.py`；`dashboard.py` 改从 `.webdata` 导入；折叠其中已证明不可达的 `try/except ImportError` 相对导入 fallback；删 `nvidb/.streamlit/config.toml`；修 `run.py` 两处 "Streamlit port" 帮助文案与 `README.md` 致谢措辞（Streamlit → Dash/Plotly）；为数据层新增 `tests/test_webdata.py`（42 例，覆盖此前零覆盖的解析/格式化/汇总函数）；手动验证 `nvidb web`（Dash 页面 HTTP 200）与 bare `nvidb --once`（真实连接远程节点）均可正常运行。**顺带修复阶段 1/2 引入的两个真实回归**：① `run.py` 顶部误删了 `from ..connection import NVClientPool`（当时以为只有死代码在用，实际上是裸 `nvidb` 默认动作在用，会导致主入口直接 `NameError` 崩溃）；② `monitor.py::_take_snapshot` 在阶段 1 删除局部 `snapshots` 字典构建时漏删了末尾 `return snapshots`，导致 `NameError`，会静默杀死 `@nvidb.monitor` 装饰器的后台采样线程。两者都不在 pytest 覆盖范围内（`nvidb/test/run.py` 的裸命令路径和 `monitor.py` 均无测试），是手动跑 CLI/装饰器验证时发现的 |
>
> ### 下一步：阶段 4（§3.1/§3.3，未开始）
>
> 共享助手收敛（truncate/display_width/阈值配色/设置合并；`tui_theme` 成为唯一家，
> `sched/model.py` re-export 保测试；executor.py 的 kill-guard shell 模板四处复制 →
> 一个模板生成器【正确性风险项，优先】）。
>
> ### 其余阶段（按 §9 顺序，未开始）
>
> - 阶段 5（§5，性价比最高）：DCGM 失败负缓存、`cpu_cores` 每客户端缓存、
>   lease 续约节流（ttl/3）、`sync_nodes_from_config` 双跑去重、keeper.status 缓存 ~10s、
>   `sched/tui.py` 主循环 `set_log_request` 守卫、dashboard 单遍聚合。
> - 阶段 6（§6.1/§6.3）：`nvidb/test/` → `nvidb/cli/`（改 pyproject 入口、
>   tests/test_ssh_proxyjump.py:18 唯一真实导入方）；`test.sh` 改跑 pytest；CI 加测试任务；
>   Python 下限决策（建议 bump >=3.9：utils.py:87 / data_modules.py:94 的 PEP 585 求值注解
>   与 connection.py 的 str.removeprefix 在 3.8 必炸）。
> - 阶段 7/8（§4）：函数拆分与 connection.py 包拆分——放在删除之后做，避免拆即将删除的代码。
> - 阶段 9（§8，逐项单独 commit 的行为变更）：utils.py import 期 `logging.basicConfig(INFO)`
>   删除（CLI 更安静，代码本意如此）；CJK 截断修复（两个 len() trim 闭包 → `tui_theme.fit`）；
>   颜色统一（connection.py `dark_grey` → `bright_black`）；裸 except 收窄。
>
> ### 本轮执行中的经验（供后续会话参考）
>
> - 计划行号已因多次前置 commit 偏移，按符号名定位；
> - `grep` 验证死符号时不要把定义文件本身从结果里过滤掉（`ServerListInfo.to_dict`
>   因此误删过一次，已恢复）；
> - 用 python 脚本按 marker 切块删除时，注意"空行"可能是 4 空格行、装饰器行在
>   def 之上（`_throughput_kib_per_second` 的 `@staticmethod` 曾被漏掉）；
> - 测试 fixture `_pool()` 以 `__new__` 构造并手工设 ~60 个属性，凡给
>  NVClientPool 新增 `__init__` 属性，需同步 fixture；
> - **`pytest tests/` 全绿不等于没有回归**：`nvidb/test/run.py` 的裸命令默认动作
>   （`nvidb` 不带子命令）和整个 `monitor.py` 都不在 pytest 覆盖范围内，阶段 1/2 在
>   这两处各留下一个 `NameError`（删导入删多了 / 删代码删漏了 return 语句），直到
>   阶段 3 手动跑 `nvidb --once` 和 `@nvidb.monitor` 才暴露。**删除"未使用的导入/变量"
>   或"死代码分支"后，必须真正跑一次 CLI 默认路径和无测试覆盖的模块，不能只看 pytest
>   和 ruff 是否通过**——ruff 对前者报的是 F401（看起来人畜无害），对后者完全没报（脚本
>   里改了函数体，ruff 不知道调用方期望什么返回值）。

Scan date: 2026-08-16, against the current working tree (including the uncommitted
changes to `connection.py`, `sched/tui.py`, and the new `tui_theme.py`).
Goal: shrink the codebase, remove dead code and unreachable fallback logic, and
improve readability/efficiency — **with zero behavior change**. Items that would
change observable behavior are collected in §8 and require an explicit opt-in.

Codebase size (excluding `build/`, which is a stale packaging copy): ~30k lines of
source. Headline: **roughly 2,500–3,000 lines can be deleted outright** with no
functional impact, the largest single item being the unreachable Streamlit UI in
`web.py` (~1,320 lines).

A useful signal from linting: `ruff` reports ~57 standing findings, **all outside
`nvidb/sched/`** (which passes clean). The scheduler package is actively
maintained; the older top-level modules (`connection.py`, `web.py`, `utils.py`,
`data_modules.py`, `nvidb/test/run.py`) are where nearly all of the cruft lives.

---

## 1. Dead code — safe deletions

### 1.1 `web.py`: the entire Streamlit UI is unreachable (~1,320 of 1,979 lines)

The CLI routes both `nvidb web` and the deprecated `nvidb log web` exclusively to
`dashboard.run_dash_app` (`nvidb/test/run.py:1033,1047`). Nothing calls the
Streamlit half of `web.py`; `streamlit`/`altair` are not declared in
`pyproject.toml` or `requirements.txt`; there are no tests. Dead ranges include
`show_live_dashboard` (992–1210), `show_logs_dashboard` (1454–1874), `main` /
`run_streamlit_app` (1877–1979), all `_render_*` / `_apply_*` / caching helpers,
and `_LiveStatsCache` (881–976).

- Keep only the ~660-line data layer that `dashboard.py:51-93` imports (parsers,
  formatters, `load_sessions` / `load_session_logs`, `_server_summary`,
  `_LOG_METRICS`, `_downsample_per_gpu`, `_build_log_snapshot_table`, `_user_*`).
- Rename the survivor to something honest (e.g. `webdata.py`), fix the docstring,
  drop then-unused imports (`argparse`, `importlib.util`, `platform`,
  `threading`, `time`).
- Also delete `nvidb/.streamlit/config.toml` (exists only for the dead
  `streamlit run` path) and fix stale wording: `run.py:989,992` ("Streamlit
  port"), `README.md:840`.
- Bonus: this dissolves ~280 lines of web↔dashboard duplication (cache class,
  server summary, GPU label map, palettes) without any merge work — the dead
  side was the duplicate.

### 1.2 `connection.py` dead symbols

| Lines | Symbol | Note |
|---|---|---|
| 71–79 | `nvidbInit()` | Only user of the top-level `import pynvml` (line 18) — both go |
| 1007–1040 | `BaseClient.format_process_summary_xml` | Never called |
| 1780–1781 | `NVClientPool.test()` | Body is `pass` |
| 1783–1810 | `execute_command` / `execute_command_parse` | Only referenced by a commented-out line; `execute_command_parse` is latently broken (param `type` shadows the builtin → `TypeError` at line 1809) |
| 1812–1832 | `get_all_system_stats` | Never called |
| 238–265 | `BaseClient.test` / `get_os_info` / `get_gpu_stats` | Referenced only by dead functions in `run.py` |
| 839–841, 1564–1565 | `get_client()` (both) | Never called |
| 117–140 | `BaseClient.query_nvml_snapshot` | Overridden by every subclass incl. test fakes; unreachable |
| 21, 24, 44, 52 | Unused imports (`AutoAddPolicy`, `cprint`, `num_from_str`, `get_memory_color`) | ruff F401 |
| 5747/5753, 6082 | `cpu_cores_digits` computed, never used | Includes a wasted `max()` pass in `print_stats` |
| 809–817 | Dead first pass of swap parsing (result discarded, re-parsed below) | 9 lines |
| 1605–1607 | `except CalledProcessError` after `subprocess.run(check=False)` | Unreachable |
| 1922–1924 | "backward compatibility" tuple-shape fallback | `get_full_gpu_info` always returns a 2-tuple; same dead shape copy-pasted at `logger.py:204-209` |

### 1.3 `nvidb/test/run.py` (the CLI entrypoint — see §6.1 for the rename)

- 931–949: `test_connection` / `test_get_os_os_info` / `test_get_gpu_stats` /
  `test_get_all_stats` — zero references; carry 5 commented-out calls.
- 15: module global `cli` — consumed only by the dead functions above.
- 249: `global test_server, cli` — `test_server` is never assigned anywhere.
- 261–262: `_format_servers_yaml` / `_format_config_yaml` back-compat aliases — zero references.
- 6: `import re` unused; 970/973: `add_parser` / `info_parser` assigned, never used.
- 343–352: a `ServerInfo(...)` constructed with 8 kwargs and discarded.

### 1.4 `utils.py` (~90 of 324 lines) and `data_modules.py` (~90 of 195 lines)

- `utils.py`: `get_gpu_memory` (16–25), `get_gpu_stats_query` (29–74, also
  structurally broken — logs and returns `None` inside `except: pass`),
  `get_memory_ratio_color` (153–183), plus `num_from_str` / `get_memory_color`
  which are imported by `connection.py` but never called. Cascade: removing these
  drops `pandas`, `subprocess`, `os`, `ET` from `utils.py`'s imports — a
  measurable `import nvidb` startup win since `utils` is imported first.
- `data_modules.py`: `GPUProcess` (48–75) and `GPUInfo` (78–94) are never
  constructed; `ServerListInfo.remove_server` (123–124), `to_yaml`+`to_dict`
  (192–195, 141–150) are never called; lines 19–45 are a 26-line commented-out
  XML block that is a verbatim duplicate of the dead `utils.get_gpu_stats_query`
  body; 9 unused imports include `paramiko` and `pandas` — both currently pulled
  into every `import nvidb`.

### 1.5 `nvidb/sched/` dead code

| Location | What |
|---|---|
| `transport.py:274-310` | `class TransportPool` — never used; `Scheduler._backends` already fills this role |
| `transport.py:51-69, 81-84, 29-31` | `Transport.write_file` + its private helpers `_dirname`, `CommandResult.ok` — no callers |
| `transport.py:238-244` | `SSHTransport.connected` property — no callers |
| `db.py:531-535` | `purge_alerts()` — no caller, no CLI command |
| `model.py:38, 40, 44-60` | `SUCCESS_JOB_STATES`, `NODE_STATES`, `EVENT_KINDS` — never read |
| `executor.py:36, 40-41` | `LaunchResult.session_isolated` / `.stdout` / `.stderr` — written, never read |

Dead parameters (all call sites use the default or always pass the same value):
`cli.py:133` `_resolve_ids(scheduler=…)`, `cli.py:114` `_maybe_tick(force=)`,
`scheduler.py:595` `tick(dispatch=)`, `scheduler.py:1023` `_terminate(grace=)`,
`scheduler.py:712/1007` optional `summary=` (always passed), `db.py:319`
`get_meta(default=)`, `db.py:860` `purge_jobs(before_id=)` (or expose it as
`job purge --before ID` — currently useful-but-unreachable),
`executor.py:284` `probe(want_process_table=)`.

Decide explicitly on `sched/__init__.py:27-56` `Queue`/`open_queue()`: it is an
intentionally exported Python API with zero users and zero tests. Keep + add one
test, or remove — don't leave it undecided.

### 1.6 Misc small dead code

- `tui_theme.py:153-155` `DiffScreen.reset()` — never called.
- `mouse.py:27-28` `BUTTON_MIDDLE` / `BUTTON_RIGHT` — never read.
- `monitor.py:47,219` `GPUStats.memory_min` — computed, never surfaced.
- `monitor.py:138-172` `_take_snapshot` returns a dict all 3 callers discard.
- `config.py:44-46` `ensure_working_dir()` — zero references.
- `logger.py:9` unused import; `:210` unused `system_info` unpack; `import sqlite3`
  repeated inside 4 methods (hoist to module level).
- `dashboard.py:11` `math`, `:42` `config`, `:81` `_format_mib` — unused imports.
- `__init__.py:6` redundant `from . import monitor` (rebound at line 9);
  `:12` `nvidb_test` alias — zero references. Add `__all__` for the intentional
  re-exports.
- `example_interactive.py` — orphan (unreferenced anywhere), demonstrates a
  degraded version of the bare `nvidb` command. Delete or fold a 5-line snippet
  into README's Python API section.
- `nvidb/src/` — contains **only** `__pycache__` with 2024-era `.pyc` files, no
  sources. `rm -rf`.

---

## 2. Unreachable / excessive fallback logic

### 2.1 The big one: ~145 pointless `getattr(self, "attr", default)` in `connection.py`

147 `getattr(self, …)` calls were enumerated and matched against `__init__`:
**every target attribute is unconditionally assigned** in `NVClientPool.__init__`
(1698–1769) or `BaseClient.__init__` (85–92); only `_tui_diff_screen` (5017) is
genuinely lazy. Highest-frequency offenders: `unified_show_trends` (15×),
`unified_active_pane` (15×), `display_mode` (14×). Includes the deceptive
5×-duplicated `bool(getattr(self, "unified_process_sort_descending", mode != "command"))`
whose computed default is dead. Mechanical replacement with plain attribute
access; also collapses idioms like `max(0, int(getattr(self, "_unified_gpu_count", 0) or 0))`.

Same family: `hasattr` guards on `__init__`-set attributes at 3985–3990,
4022–4023, 4031–4032, 1777–1778.

### 2.2 Nine unreachable import fallbacks in `web.py` / `dashboard.py`

`try: from nvidb.X import … except ImportError: from .X import …` at
`web.py:33,37,849,863,1214` and `dashboard.py:40,44,50,382`. The relative branch
is provably unreachable (if the absolute import fails, `__package__` is unset and
the relative import fails too — the `# pragma: no cover` on each is the
admission). `dashboard.py:50-90` writes 21 imported names out twice, 40 lines.
Collapse all nine to plain relative imports (~70 lines removed).

Keep the *genuinely reachable* guards: `dashboard.py:20-35` (`dash` is not a
declared dependency), `web.py:22-24` (`streamlit` — moot after §1.1),
`nvml.py:53,383` (`import pwd` on non-POSIX).

### 2.3 `monitor.py:16-20` `PYNVML_AVAILABLE` can never be `False`

`import nvidb` → `connection` → `nvml` does a hard `import pynvml`, and
`nvidia-ml-py` is a hard dependency. Either delete the flag or stop importing
`monitor` eagerly from `__init__.py` and make the guard honest.

### 2.4 `nvml.py`: triple-layered defense around guaranteed symbols

`getattr(pynvml, "…", None)` chains + `_safe`'s exception swallow + `callable()`
re-check for symbols that ship in every pynvml release (79–85, 94–99, 111–132,
139–155, 220–271). The constant fallbacks are pure noise
(`getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)` — the default equals the real
value). Reference `pynvml.nvmlX` directly; keep `_safe` only for calls that
genuinely vary by driver/GPU (fan speed, architecture, PCIe gen/throughput, MIG).

### 2.5 Unreachable `except` clauses and provable no-ops in `connection.py`

- 1535–1542: two `except` clauses with character-identical bodies — merge.
- 1527–1530, 1216–1221: `except` around callees that already swallow everything.
- 664–676, 5847–5856: `try/except` around pure dict/list operations that cannot raise.
- No-op conditionals: 1938 (`stats.copy() if not stats.empty` inside
  `if stats.empty:`), 2054, 1955–1956 (`x[:11] if len(x)>11 else x` ≡ `x[:11]`),
  5037, 105, 5842–5843, 5567–5570.
- Redundant isinstance guards on internally-produced data: 2553–2557, 2160,
  2534–2547, 4534, 4574, 120–125.
- 15 `except Exception: pass` sites; the worst is `_keyboard_listener:7063-7064`
  which swallows every exception from the whole input loop and hides real bugs.

### 2.6 `sched/` defensive trims (smaller, the package is clean overall)

- `model.py` `from_row` methods coerce columns the schema already declares
  `NOT NULL DEFAULT …` (262–294, 530–539, 443–453) — trim to the genuinely
  nullable set.
- `scheduler.py:218,266,291` `(self.cfg or {})` — `__init__` guarantees a dict.
- `executor.py:483-485` order-defensive carry that the emitter's fixed ordering
  makes unreachable; `executor.py:293` double-filtering already-filtered specs.
- Bare `except:` → narrow: `transport.py:151`, `cli.py:52`, `run.py:743,750,817,884`,
  `monitor.py:134,151` (bare excepts also swallow `KeyboardInterrupt`).
- `tui.py:179-181`: config errors at worker startup are silently discarded while
  `self.error` exists precisely to report them — record instead of `pass`.
- `dcgm.py:13-19`: the port-less TCP fallback candidates are behaviorally
  identical to the first two (pydcgm appends the default port) and can never
  succeed where those failed; `dcgm.py:159-164` shares one `except: pass`
  between `WatchFields` and `UpdateAllFields`, so a real update failure still
  reports `"ok": True` — split them.
- `metrics.py:12-13,47-50`: `if df_columns is None` guards no call site can hit.

---

## 3. Duplication to consolidate

### 3.1 Repo-wide helper sprawl (one implementation each, please)

| Concern | Copies today |
|---|---|
| Truncate-to-width | 5+: `tui_theme.fit`, `sched/model.fit_display`, `sched/tui._fit` (pure alias), 2 closures in `connection.py` (2883, 4701 — these use `len()` not display width, so CJK shears those panels; fixing is a behavior change, see §8), plus 4 more ad-hoc `x[:w-3]+"..."` sites |
| `display_width` | 2: `tui_theme.py:34-43` and `sched/model.py:137-151` — identical logic |
| Byte/MiB formatting | 5: `monitor._format_bytes`, `web._format_mib`, `web._format_gb`, `sched/model.format_mb`, a `connection.py` closure — four different unit conventions (unifying output strings is a behavior change; unify implementation, keep per-caller format) |
| Threshold→color | 4 in `connection.py` (2675, 4194, 4174, 5377) + `utils.get_pcie_load_color` — same `>= critical → red, >= warning → yellow` shape |
| "First number in a string" | 3 in `connection.py` (58, 2186, 5343) — all reducible to `extract_numbers(...)[0]` |
| used/total-ratio parsing | 3 in `connection.py` (2200, 4176, 5357) + `web.py:135` + `dashboard.py:450` |
| `os.get_terminal_size()` try/except | 10 sites in `connection.py` with three different fallbacks — one `_terminal_size(fallback)` helper |

**`tui_theme.py` extraction is half-finished** — it exports `display_width`/`fit`
but nobody imports them from there, and its `MUTED` token exists while
`sched/tui.py` hard-codes `"bright_black"` ~45 times and `connection.py` uses
`"dark_grey"` for the same role. Direction: make `tui_theme` the single home;
have `sched/model.py` re-export (keeps tests importing from `sched.model` green);
adopting `MUTED` in `sched/tui.py` is behavior-identical, in `connection.py` it
changes a color (§8).

### 3.2 `nvml.py`: local collector vs embedded remote agent (~120 duplicated lines)

`PynvmlCollector` and the `Nvml` class inside `REMOTE_NVML_AGENT_SCRIPT` are two
implementations of one data model: byte-identical `_ARCHITECTURE_NAMES` tables,
identical decode/cuda-version/username/process-name helpers, identical
slow-metric TTL logic, and — the real risk — an **identical 24-key GPU dict
schema** maintained by hand in both places. `dcgm.py:47,62,70` already shows the
fix used in this repo: inject shared constants into the generated script with
`!r` formatting. Do the same for the architecture table and the schema key list.

### 3.3 `sched/`: cli.py ↔ tui.py render duplication

- `_gpu_process_lines` exists in both (`cli.py:149-201`, `tui.py:781-834`) with
  the same blind-mode special case and unmanaged-only filter, but different
  truncation constants (3 vs 2) under two names.
- Byte-identical `source = "blind" if … else f"{n}p"` (`cli.py:221`, `tui.py:733`).
- Keeper up/DOWN badge in 3 places; job row cell formatting rules in 2 places;
  terminal-state literals in 4 places (`model.TERMINAL_JOB_STATES` vs literal
  tuples in `tui.py:49,1213` and `cli.py:41-43` — these drift silently if a
  state is added).
- Settings normalization written 3× with the same shape (`scheduler.load_settings`,
  `notify.load_notify_settings`, `backup.load_settings`) → one `merge_settings`.
- `backup.py:65-82` re-implements `model.utcnow`/`parse_ts` (only adds trailing-`Z`
  handling — extend `model.parse_ts` instead).
- `cli.py:543` reads `notifier._desktop_command` across the boundary → add
  `Notifier.active_channels()`.
- **Safety-critical**: the `kill -0 && ps … | grep -F -q -- "$d/run.sh"`
  "is this pid still my job" shell guard is copy-pasted 4× in `executor.py`
  (221, 304, 354, 362). One template builder; four copies of this particular
  string is a correctness risk, not just noise.
- Keeper liveness matching exists in Python (`keeper.py:292-330`) and shell
  (`keeper.sh:35-51,165-173`) — can't be deduplicated, but the match rule should
  be one documented constant referenced by both, plus a parity test.

### 3.4 `connection.py` internal duplication (largest items)

- `print_stats:5814-5838` vs `_background_refresh:7093-7113` — 21 byte-identical
  lines (fetch → record history → cache write) → `_refresh_cache()`.
- `connect()` auto-branch 1420–1470 vs key-branch 1471–1521 — ~40 of 51 lines
  identical → `_try_key_auth(...)`.
- `_format_unified_process_details:3476-3572` builds `usage_fields` fully, then
  discards and rebuilds it for compact layout — 46 lines.
- The 4-line process-selection reset appears at ~10 sites while the helper
  `_reset_unified_process_filter_selection` already exists — use it.
- Key bindings are described in 3 places (help panel, controls line,
  `print_refresh` handler comments) — three sources of truth.
- Plus ~15 more localized pairs (mouse-event branches, action-notice block,
  pagination/selection resolution, server-summary dict tails, `safe_get_text`
  defined twice, …) — see the per-line list in §1.2's audit; all are
  extract-local-helper fixes.
- "Is this a compact screen" (`terminal_height < 28` / `< 36` + trends) is
  re-derived independently at 4 sites — compute once per frame.

### 3.5 `dashboard.py` internal

- Percent-helper columns byte-identical at 631–634 and 942–945.
- Horizontal-bar figure recipe duplicated (`_user_vram_fig` 652–670 vs
  `_time_share_fig` 1199–1216).
- Theme values written twice: Python `_PALETTES` dict vs the 178-line CSS string
  (same hex colors in both syntaxes) — generate the CSS variable block from
  `_PALETTES`.

---

## 4. Oversized functions / module split

### 4.1 `connection.py` (7,239 lines) → package split

`NVClientPool` alone is 5,627 lines / 91 methods, but its line ranges already
form clean seams. Proposed behavior-preserving split (file moves + mixins so the
public surface and tests keep working):

| New module | From lines | Content |
|---|---|---|
| `clients/base.py`, `clients/remote.py`, `clients/local.py` | 82–1610 | The three client classes |
| `collect.py` | 1834–2138 | DataFrame shaping, PCIe columns |
| `unified_model.py` | 2140–2530 | Table build/filter/sort/paginate/capacity |
| `render/process_pane.py` | 2687–3905 | 1,219 lines |
| `history.py` | 3907–4107 | Sparklines/trends |
| `render/table.py`, `render/cards.py`, `render/help.py` | 4109–5555 | 1,447 lines |
| `summaries.py` | 5557–5802 | Per-node summaries |
| `tui_state.py` | 6117–6703 | View-state mutators |
| `tui_input.py` | 6705–7064 | Key/mouse handling |
| `app.py` | 5804–6115, 7066–7239 | Frame assembly + loops |

Functions worth splitting regardless of the module split:
`_format_unified_process_details` (**1,078 lines**, 12 nested closures),
`_format_fixed_width_table` (473 — column selection / width solving / rendering
are three independent phases), `get_full_gpu_info` (418 — promote its closures to
methods, extract `_fetch_dcgm_advanced`), `_format_unified_detailed_table` (410),
`print_stats` (312), `_render_unified_gpu_lines` (253), `_handle_keypress` (192 —
flat if-ladder repeating the same guard 5×; use a dispatch dict). Name the bare
terminal-size magic numbers (28/36/70/56/…).

### 4.2 `sched/` splits

- `cli.py:1254-1564` `register_parsers` (311 lines) → `_register_queue_parsers` +
  `_register_job_parsers`; drain/resume + ignore/unignore blocks become one loop.
- `scheduler.py` `_dispatch` (164), `_reconcile_jobs` (148), `_build_gpu_states`
  (116) → extract `_build_budgets`, `_fail_job` (two 18-line near-identical
  blocks), `_try_launch`, `_settle_finished`, `_retry_vanished`, `_mark_lost`,
  `_attribute_processes`. The vanished-job reset field list (822–838) duplicates
  `requeue()` (567–587) → shared `_clear_placement_fields()`.
- `tui.py` `_frame_lines` (150 — job pane written twice for the detail/collapsed
  branches), `_footer_lines` (136 — separate control-list building from layout),
  `_job_lines` (111), `_activate` (102 — 20-branch if/elif → dispatch table).
- `cli.py`: 21 handlers repeat `scheduler = _open(args); try: … finally: close()`
  → one `with _scheduler(args) as s:` context manager; `if args.json: … else:
  print(…)` appears 27× → `_emit(args, payload, text)`.
- `executor.py:177-278` `launch`: move the 30-line shell bootstrap into a
  module-level template (it's shell, not Python — `keeper.sh` sets the precedent).

### 4.3 `dashboard.py`

`create_app` (252 lines) → `_build_layout` + `_register_live_callbacks` +
`_register_logs_callbacks` + `_install_stale_callback_handler`;
`_live_update_payload` (149) and `_render_session_body` (148) → extract the
card/section builders. The 13 parallel lists in `payload` encode a positional
contract with `_build_live_skeleton` that nothing enforces — document or use a
small dataclass.

---

## 5. Efficiency (behavior-identical fixes)

### 5.1 Per-tick remote round-trips (the ones that actually cost wall-clock)

- **DCGM probe never negative-caches** (`connection.py:582-683`): the snapshot
  command runs on every host on every refresh forever, even after failing
  hundreds of times — one extra SSH exec + remote Python startup per host per
  second on clusters without DCGM. Cache the failure per client.
- **`cpu_cores` re-queried every tick** (`connection.py:686-837`): `nproc` is
  static; cache per client. Also consider merging the 3 Linux stat commands into one.
- **Scheduler probes make 2 SSH round trips per node per tick**
  (`scheduler.py:683-689`): NVML agent + executor probe are both plain shell —
  concatenate with a marker (output is already marker-delimited). Halves per-tick
  SSH latency.
- **`keeper.status()` inside `snapshot()`** (`scheduler.py:1532`): spawns 1–2
  processes per TUI refresh (default 3 s) forever, to draw a 4-char badge. Cache
  ~10 s.
- **`logger.py:230-235`** calls `get_process_summary` once per GPU row — one `ps`
  round-trip per GPU instead of the per-host batch `connection.py` already does.
- `_failure_detail` makes up to 2 remote reads per failed job → one command.

### 5.2 Redundant recomputation per frame (`connection.py`)

- `_build_unified_gpu_table` (full `pd.concat`) runs 2× per refresh + once more
  per keypress → memoize per tick.
- `_unified_gpu_capacity` regex-parses each row ~6× per frame (filter, sort,
  summary, band, history, table) → compute once into hidden columns at build time.
- `_unified_section_headers` rescans the whole table per node (O(nodes×rows)) →
  one `groupby`.
- `_client_table_identity` calls `socket.gethostname()` O(rows×nodes) times per
  refresh → build the identity map once.
- 5 separate `iterrows()` passes in `get_client_gpus_info` → one.
- `print_stats:6098` `_screen_line_count` inside the per-server loop is O(n²) in
  rendered lines → running counter.
- `pd.set_option` ×3 per refresh → module import time.

### 5.3 Scheduler DB traffic (`sched/`)

- **Lease renewal**: `_renew_tick_lease()` is called from 16 sites, mostly inside
  per-item loops — ~45 write transactions per tick on a modest queue. Throttle by
  time (`ttl/3`); identical guarantees, ~10× fewer writes.
- `sync_nodes_from_config()` runs twice per CLI invocation (`_open` + first
  `tick`), each an `upsert_node` write per node — drop the `_open` call or batch.
- Nodes fetched twice per tick; `live_jobs` fetched twice per node and twice
  globally per tick — fetch once and thread through.
- `snapshot()` fetches 20 jobs to keep 10 (push the filter into SQL);
  `list_alerts` + `open_alert_count` → one query.
- `init_schema` runs `executescript` + one `PRAGMA table_info` per table on every
  `open_db` → skip when `meta.schema_version` already matches.
- `_wait_for` polls one `get_job` per id per 2 s → one `WHERE id IN (…)`.
- `_impossible_reason` is O(pending × nodes × gpus) per tick for an answer that
  only changes when inventory changes → precompute per tick.

### 5.4 Dashboard / NVML

- `dashboard.py`: the same server table is walked 3× per refresh per server
  (`_server_summary` + two `iterrows()` loops computing the same aggregates) →
  one pass. `cache.snapshot()` copies the full 360-point history twice per tick,
  once for a signature that never reads it → cheap `raw_snapshot()`.
  `_live_history_figs` rescans the entire history per server per tick →
  bucket once (or maintain per-server deques in `_record_history`).
  `_gpu_datatable` calls `to_dict("records")` twice on the same frame.
  The collapsed "Raw data" table serializes the entire log frame every render.
- `nvml.py`: the remote agent re-resolves ctypes symbols and re-assigns
  `argtypes`/`restype` on every call (~12 rebinds per GPU per snapshot) → bind
  once in `__init__`. `_memory_info` re-attempts the v2 API per GPU per snapshot
  after it has already failed → probe once. `_get_processes` does per-tick
  `/proc` + `pwd` lookups and a redundant `nvmlSystemGetProcessName` per
  iteration for pids already seen → pid-keyed cache (same pattern exists in the
  remote agent).

### 5.5 TUI

- `sched/tui.py:1847-1851`: `worker.set_log_request(...)` is called every frame
  (~2.5/s), taking the state lock each time — move the call behind a
  selection/visibility change check (the `!=` guard already exists inside).

---

## 6. Packaging / repo hygiene

### 6.1 `nvidb/test/` is the CLI, not tests

`pyproject.toml:46` — `nvidb = "nvidb.test.run:main"`. It has zero overlap with
`tests/` (the real pytest suite). `test.sh` runs the **CLI**, wrapped in
`print("Running test")`/`print("Test complete")`. Neither CI workflow runs pytest.

→ Rename `nvidb/test/` → `nvidb/cli/`; update `pyproject.toml`, `__init__.py`,
and `tests/test_ssh_proxyjump.py:18` (the one real importer); keep a shim module
for one release if PyPI back-compat matters. Replace `test.sh` with `pytest` and
add a test job to CI. Delete the two misleading prints.

### 6.2 Stale `build/` can ship a broken wheel

`build/lib/` is gitignored but stale on disk (predates the last 4 commits;
`connection.py` is 405 lines behind) and — critically — **is missing
`tui_theme.py` entirely**, while `VERSION` is `1.8.0` in both places. A local
`python -m build` without cleaning could ship a 1.8.0 wheel whose
`connection.py:41` import of `tui_theme` fails on install. `rm -rf build/
nvidb.egg-info/ nvidb/src/` now; clean before local builds (CI runners are fresh,
so CI is safe).

### 6.3 The declared Python floor is false

`requires-python = ">=3.8"`, but the package cannot import on 3.8:
`utils.py:87` and `data_modules.py:94` use PEP 585 `tuple[...]`/`list[...]` in
evaluated annotations without `from __future__ import annotations`, and
`connection.py:4853` uses `str.removeprefix` (3.9+). Either bump to `>=3.9` and
drop the 3.8 classifier (recommended — matches reality), or add the two
`__future__` imports and replace `removeprefix`.

### 6.4 Two real bugs found during the scan (fix or delete alongside)

1. `run.py:246-258` `init()`: opens and parses the config YAML then throws the
   result away (re-parsed by `from_yaml` on the next line); the unguarded `open`
   turns a missing config into a raw traceback instead of the friendly "Run
   'nvidb add'" message; and `RemoteClient(server_list[0])` IndexErrors on an
   empty `servers:` list while building an object only dead code reads. Reduce to
   `return ServerListInfo.from_yaml(config_path)`.
2. `connection.py:1792-1810` `execute_command_parse`: the `type` parameter
   shadows the builtin so line 1809 would raise `TypeError` if ever called —
   confirmation it is dead (§1.2); delete rather than fix.

---

## 7. Test coverage map (what guards the refactor)

Well covered — refactor freely, run the suite between steps:
- `sched/*`: `tests/test_sched_*.py` (~3,300 lines) drives the CLI end-to-end,
  asserts on TUI render output, and `fake_cluster.py` mirrors the `JobExecutor`
  interface (**mirror any `executor.py` signature change there**, 186–254).
- `connection.py` rendering/reconnect: `test_tui_views.py` (1,865 lines),
  `test_pcie_link_load.py`, `test_connection_reconnect.py`, `test_nvml_collection.py`.
- `ssh_proxy.py`, `config.py`, `mouse.py`: dedicated suites. `ssh_proxy.py` is
  clean — leave it alone.
- `skills/nvidb-queue/SKILL.md` is enforced by `test_skill_commands.py` — CLI
  surface changes will be caught.

**Not covered at all**: `web.py`, `dashboard.py`, `monitor.py`, `dcgm.py`,
`tui_theme.py`. For the §1.1 deletion and any `dashboard.py` work, add cheap
characterization tests first (e.g. `_server_summary` output on a fixed frame,
`PynvmlCollector.collect()` against a fake pynvml) — otherwise "behavior
unchanged" is unverifiable there. Verify `nvidb web` by hand after §1.1.

---

## 8. Items that are NOT behavior-preserving (opt-in, separate commits)

Everything above is intended to keep observable behavior identical. These are
worth doing but change something visible — do them consciously, not as part of
the cleanup:

1. **Logging fix**: `utils.py:9` runs `logging.basicConfig(level=INFO)` at import
   time, which makes the CLI's own `basicConfig(WARNING)` in `main()` a no-op and
   hijacks root-logger config for any library consumer. Deleting it makes the CLI
   quieter (what the code already intends) — still, output changes.
2. **CJK truncation fix**: replacing the two `len()`-based `trim` closures in
   `connection.py` with `tui_theme.fit` fixes panel shearing on wide characters
   and changes the ellipsis glyph.
3. **Color unification**: adopting `tui_theme.MUTED` in `connection.py` changes
   `"dark_grey"` → `"bright_black"` chrome.
4. **Python floor bump** to `>=3.9` (or keep 3.8 and fix the three breaks).
5. **Byte-formatting unification** (§3.1) if output strings are normalized.
6. Narrowing bare `except:` clauses technically changes behavior on
   `KeyboardInterrupt` during those windows — in every case for the better.

---

## 9. Suggested execution order

Each phase is independently committable; run `pytest` after every step.

1. **Zero-risk deletions** — `rm -rf build/ nvidb.egg-info/ nvidb/src/`; delete
   `example_interactive.py`; dead functions/aliases in `run.py`, `utils.py`,
   `data_modules.py`, `config.py`, `connection.py` (§1.2), `sched` dead code
   (§1.5); `ruff check --select F401,F841 --fix` for unused imports/vars.
2. **Unreachable fallbacks** — the 9 import fallbacks (§2.2), the ~145
   `getattr`/`hasattr` in `connection.py` (§2.1), no-op conditionals and
   unreachable excepts (§2.5), `PYNVML_AVAILABLE` (§2.3), `run.py` `init()` fix (§6.4).
3. **Delete the Streamlit UI** (§1.1) — own commit, manual `nvidb web` check.
4. **Shared helpers** (§3.1, §3.3): truncation/display-width/threshold-color/
   settings-merge consolidation; finish the `tui_theme` extraction via
   re-exports; the `executor.py` shell-guard template.
5. **Cheap efficiency wins** (§5): DCGM negative cache, `cpu_cores` cache, lease
   throttling, double `sync_nodes_from_config`, keeper-status cache,
   `set_log_request` guard, dashboard single-pass aggregates.
6. **Packaging** (§6.1, §6.3): `nvidb/test/` → `nvidb/cli/`, pytest in CI,
   version floor decision.
7. **Function splits** (§4) — after the deletions, so you aren't splitting code
   you were about to remove.
8. **`connection.py` module split** (§4.1) — last and largest; the render suites
   in `tests/` pin its output.
9. **Opt-in behavior changes** (§8) — each its own reviewed commit.

Estimated net effect of phases 1–3 alone: **~2,800 lines removed**, `import
nvidb` no longer pulls `pandas`+`paramiko` via `data_modules`/`utils` dead paths,
and every remaining line is reachable.
