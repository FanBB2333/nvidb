# nvidb 下一版本改造方案：自治 lane 与证据驱动的对账

> 状态：阶段 A 已完成并启用（v1.9.3）；阶段 B/C 待实施。方案基于 `main@4ea470c`（v1.9.2）的代码扫描与
> `~/.nvidb/queue.db` 中 2026-09-21/22 夜间的真实事件写成。
> 代码位置一律用 `文件::符号` 标注，行号会漂移。

> **实施更新（2026-09-22）**：本次实现阶段 A，并以保留现有队列为约束：
> 保留任务 ID、attempt、依赖、lane 顺序、数字并发与 lookahead；默认仍为 1。
> 阶段 B/C 尚未实现，`all`、`auto`、ready-first 和候选 lane 均未启用。
> `lane.json` 与超时标记属于兼容扩展；超时终态要求节点结果或停止证据。
> runner 替换先等待停止派发确认，保留作业进程；同步与 claim 互斥，防止重复执行。
> 以下正文保留原始设计，实施边界以 README 的升级说明为准。
> 验证：727 项测试通过，Ruff 与 Python 3.9 语法检查通过；真实数据库副本的
> 4,429 条任务和 14 条 lane 原有字段在迁移后保持一致。已启用 macOS launchd，
> 4 条活跃 lane 已更新 runner，切换时运行任务的 attempt、run_dir 和 PID 保持一致。
>
> 复核修正：本机当前 `orphans` 已为空，事故涉及的 18 个 job 当前均为 completed。
> 历史自愈不再承诺覆盖这 18 条；只修复同一 attempt 的误判，不覆盖重跑结果。
> 三节点同时断连支持共同链路故障的推断，尚不足以定位到本机 Tailscale。
> 阶段 B 的安全回滚、attempt/revision CAS、strict 下 held 占位和跨节点依赖，
> 仍须在相应功能实现前完善，不能仅凭脚本 hash 或“只增加列”保证兼容。

> **补丁更新（1.9.4）**：修复 `cancel --no-tick` 取消未下发的 runner lane 任务时，
> 被其他节点的 SSH/tick 调度锁挡住的问题。取消与暂存通过短事务检查当前任务状态；
> 已下发、运行中、自由池和不使用 runner 的 lane 仍保留原有互斥流程。
> 新 tick 声明兼容能力，旧 tick 或管理操作持锁时仍保留保护。没有改变 runner 脚本、
> lane 设置或数据库 schema；这不是阶段 C 的完整 CAS 改造。
> 验证：734 项测试通过，包含 7 项本次回归用例；Ruff 与 Python 3.9 语法检查通过。

## 0. 事故复盘：代码里到底发生了什么

先把用户描述的那一夜（CST 2026-09-22 00:26–07:41）和数据库事件对上，
每一条都能在代码里找到对应的分支。时间为 UTC，括号内为 CST。

| 时间 | 事件（`events` 表） | 对应代码 |
|---|---|---|
| 16:32:06–07 (00:32) | 三个节点同一秒 `node_down`（406 `TimeoutError`、A100/4090 `Error reading SSH protocol banner`） | `Scheduler.tick` → `_refresh_node` 抛 `TransportError` → `_mark_node_down`。三台同时断说明是本机链路（tailscale）而非节点 |
| 16:32–23:41 | 三个节点各 lane 的 runner 只跑完手里的队头：4090 上 4295–4301 六个作业 `completed`，其余 lane 空转 | `DEFAULT_SETTINGS["lane_lookahead"]=1`；`_lane_specs` 只暂存窗口内的作业，其余顺序只存在本机 SQLite；runner 的 `maybe_start` 只能看到 spool 里的 spec |
| 17:24–20:16 (01:24–04:16) | **18 个作业被记为 `job_timeout`**（A100 8 个、4090 7 个、406 3 个，"killed after 10800s/14400s"），实际它们都已在节点上正常结束 | `_enforce_timeouts` 遍历 `dbm.live_jobs()`（不看节点状态），用 `Job.elapsed_seconds()` 的**墙钟**判超时；节点不可达时 `_terminate` 抛错 → `_remember_orphan("timed out while unreachable")`，记录仍被改成 `timeout` |
| 同上 | 9 个下游作业 `job_held`（"dependency 4245 ended as timeout" 等） | `_gate` → `_dependencies_ready` 看到上游终态非 `completed` → `_hold` |
| 23:41 (07:41) | 三节点 `node_up`；A100/4090 上一批 `job_finished completed` 补记 | `_reconcile_jobs` / `_reconcile_staged` 从 `exit_code` 文件对账（这条路是对的） |
| 23:46–23:54 | 用户手工 `requeue` 了 4201/4224/…/4292 共 13 个；紧接着 4243/4241/4246/4251/4268 自动 `job_unheld` | `_gate` 在 `hold_reason is None` 时调 `_unhold` —— 但只对"被 gate 到的作业"生效 |
| 09-22 00:22 (08:22) | 4248 仍是 held，用户手工 `job release`（会**丢掉依赖**） | 4248 在 406:0 lane 窗口之外：`_lane_specs` 在 `len(specs) >= limit` 时 `break`，窗口后的作业永远不会被 `_gate` 重新评估，`held_reason` 就一直挂着 |
| 09-19 16:26 → 09-22 00:38 | keeper 停了两天半（`queue-keeper.log`："stopped"）；期间只有交互会话在 tick | macOS 走 `nohup` 分支，与启动它的 shell 同进程组，会话结束即被 TERM；只有 `remote:` 客户端会 `ensure`，本机命令不会 |
| 09-19 12:31 | daemon 因 `SchedulerLeaseLost` 退出，keeper 5s 后重启 | `cmd_daemon` 未捕获该异常 |

另外几条在代码里直接可证的：

- **并发数热更新失效**：runner 的 `Runner.concurrency()` 先读 `self.current`
  里作业 spec 的 `concurrency` 字段，再读待运行 spec —— 并发数是随作业 spec
  下发的，改了 lane 只影响之后暂存的 spec，正在跑的旧 spec 一直占着旧值。
- **软重启卡在 stopping**：`Runner.run` 里 `if os.path.exists(STOPFILE): self.stopping = True`
  **只置位不复位**；下一 tick 的 sync 会 `rm -f "$d/stop"`，但 runner 仍是
  stopping：不 `maybe_start`，而 `self.current or self.pending_specs()` 为真又不退出
  —— 只要控制端持续暂存，就会永远卡住，只能 `--hard`。
- **cancel --no-tick 输出非 JSON**：`job cancel` 的 argparse 根本没定义
  `--no-tick`，报错来自 argparse 的纯文本 usage。
- **写操作抢锁**：`cancel` / `requeue` / `lane_skip` / `set_node_*` 都包在
  `_operation_lease` 里，与 tick 争同一把 `TICK_LOCK`；一次 tick 对 3 个节点要做
  `probe_gpus`(1) + `executor.probe`(1) + 每条 lane 一次 `lane_sync`，现在是
  2+2+8 条 lane ≈ 19 次 SSH 往返，经常超过十几秒，所以频繁 "scheduler is busy"。
- **A100 lane 名义预留**：`polyu-A100:0/1` 当前 `concurrency=4`；runner 的
  `maybe_start` 只用 `nvidia-smi` 的瞬时 `memory.free` 做门槛，刚启动、还没申请
  显存的作业不计入，于是 4 个 30G 的作业能在同一秒全部启动。

## 1. 目标与不变量

目标只有一个：**节点在、队列主机不在时，节点上的 lane 仍能按既定顺序（含依赖）
一直跑下去；主机回来后只做对账，不做任何基于猜测的状态改写。**

以下既有设计决策继续保留，不在这次改动范围内推翻：

- 不要求控制端常驻 daemon；任何客户端的 tick 都能推进队列，daemon/keeper 仍是可选。
- 按 VRAM 预算而不是整卡分配；外来进程占用要被减掉。
- lane 的打印顺序是承诺（`strict` 仍为默认）；死依赖 hold 而不是 fail。
- 节点上零安装，runner 仍以内联脚本随 SSH 下发、按内容 hash 自动滚动。

## 2. 架构上的一句话变化

现在：**控制端决定一切**（依赖、顺序、超时、启动），节点 runner 只是"帮忙点火"的手，
手里只拿一张 spec。

改后：**控制端只负责编辑与对账，节点 runner 负责执行。**

- runner 持有整条 lane 的有序 spec（含每个 spec 的依赖门控信息），自己判断
  "下一个能不能跑"、自己执行超时、自己按显存决定并发。
- 控制端每 tick 做的是：把本地数据库的"期望状态"与 runner 的"实际状态"做 diff，
  只下发差量；从节点上的**持久文件**（`exit_code` / `started` / `timeout`）读回事实。
- 控制端在没有证据时**不改任何作业的终态**。

## 3. 分项设计

优先级沿用用户列表的顺序；每项给出现状 → 改动 → 兼容 → 验收。

### 3.1 节点侧自治派发（P1）

**现状**

- `_lane_specs(lane, limit)`：`limit = max(lane_lookahead, concurrency)`，默认 1；
  遇到第一个 `_gate` 返回 `waiting/held` 的作业就 `break`。
- `runner_agent.build_sync_command`：每 tick 把窗口内所有 spec 全量重写，不在
  `keep` 里的全部删除。
- `--node X --gpus 0` 的 CPU 作业走自由池 `_dispatch` → `_launch`，只有控制端 tick 能启动。

**改动**

a. **整条 lane 下发**。`lane_lookahead` 新增取值 `all`（设为默认），`_lane_specs`
   不再在 gate 未通过时 `break`，而是把每个作业连同门控信息一起暂存：

   ```json
   {
     "job_id": 4248, "seq": 140000, "run_dir": ".../jobs/4248-1",
     "after": [
       {"job": 4245, "need": "completed", "run_dir": ".../jobs/4245-2"}
     ],
     "gated": false,
     "max_runtime_s": 14400, "vram_mb": 30000, "gpu_ids": [0],
     "script_b64": "..."
   }
   ```

   - `after[]`：**同节点**依赖，runner 直接读 `<run_dir>/exit_code` 判断
     （`need=completed` 要求退出码 0，`need=any` 只要求文件存在）。同节点不同
     lane（cpu lane 的 fetch → GPU lane 的 train）也走这条。
   - `gated`：控制端拥有的开关。依赖在别的节点、或依赖尚无 `run_dir`（自由池
     未派发）时为 `true`；控制端看到依赖到终态后改写 spec（digest 变了自然重发）。
     死依赖（hold）→ 控制端把 spec 撤回并保持 `held_reason`（策略见 3.4）。
   - 依赖被 requeue 会拿到新的 `run_dir`（attempt+1），spec 随之重写，runner 看的
     永远是正确那次 attempt 的退出码。

b. **差量同步**。runner 的 `state.json` 中 `queued` 改为
   `[{"job_id", "seq", "digest"}]`（digest = 规范化 spec JSON 的 sha1 前 12 位）。
   控制端 `_sync_lane_runners` 先算期望集合 `{文件名: digest}`，与上次读到的
   `lane.runner_state["queued"]` 比较，`build_sync_command` 只输出：新增/变更的
   写入、多余的 `rm`。runner 未上报或版本不符时退回全量。这样一条 40 个作业的
   lane 每 tick 的 SSH 载荷是 O(变化数) 而不是 O(40 × run.sh)。
   reorder = 改文件名里的 seq，一次小写入。

c. **每节点一次往返**。把 `NodeBackend.probe_gpus`、`executor.probe`、各 lane 的
   `lane_sync` 合成一条 `node_sync` 脚本（各段用 marker 分隔，沿用现有
   `PROBE_MARKER`/`STATE_MARKER` 解析方式）。4090 从 10 次往返降到 1 次。这是
   `refactor.md` §5 里已经列过的项，放在这里一起做。

d. **CPU lane**。每个节点自动建一条 `NODE:cpu` lane（`gpu_ids=[]`，
   `concurrency = max_cpu_jobs_per_node`，runner 驱动）。`--node X --gpus 0` 的作业
   在 `submit` 时直接落到 `X:cpu`；不带 `--node` 的 CPU 作业在 `_dispatch` 选好节点
   后 `lane_assign` 到该节点的 cpu lane 再暂存。不带 `--lane` 的自由池 GPU 作业
   在本阶段仍走 `_launch`（executor 直接起进程）；到 3.7 落地后自由池等价于
   `--lane any`，届时 `_launch` 只剩 `runner: off` 的 lane 一条路。

e. runner 端对应：`pending_specs()` 之外新增 `runnable_specs()`（过滤 `gated` 与
   `after` 未满足者）；`maybe_start` 按 lane 策略（3.4）从中取。

**兼容**：旧 runner 收到带 `after`/`gated` 的 spec 会忽略这两个字段直接跑，所以
`RUNNER_SCRIPT` 改动带来的 hash 变化会让控制端在第一次 sync 时自动换掉旧 runner
（现有机制：版本不符 → `kill -TERM` → 新 runner `adopt`），不需要人工干预。
`lane_lookahead` 整数值继续有效。

**验收**：拔掉本机网络 2 小时，一条 12 个作业、含同 lane `--after` 链和 cpu lane
fetch 依赖的 lane 应全部跑完；主机回来后 `job ls` 的状态与节点 `exit_code` 一致，
`events` 里没有 `job_timeout`/`job_held`。

**关于"把队列主机搬到 406"**：这是部署选择，不是代码改动，README §3.9 已支持
（406 上 `include_local: true` + 笔记本 `remote:`）。P1 落地后笔记本断线不再影响
节点执行，搬不搬只影响"断线期间能不能提交/改顺序"。建议 P1 先做；若之后仍想搬，
迁移只需 `nvidb queue backup` → scp → 406 上 `keeper install --systemd --start`。
代价是 TUI 每次刷新多一跳 SSH，且 406 变成单点。

### 3.2 重连后的对账：只凭证据改终态（P2）

**现状**：`_enforce_timeouts` 对所有 `running` 作业用墙钟判超时，节点 `down` 时照判；
判成 `timeout` 后把 pid 记进 `orphans`，节点回来时 `_reap_orphans` 只负责 kill，
不看 `exit_code`。

**改动**

a. **超时在节点上执行，控制端只读结果。**
   - lane 作业：runner 已有 `enforce_timeouts`（按自己的 `started_at` 计时），补一
     步：kill 时写 `<run_dir>/timeout` 标记文件（内容为 epoch）。
   - 自由池 / `runner: off` 作业：`executor.build_run_script` 里加看门狗
     `( sleep $N; touch "$d/timeout"; kill -TERM -- -$pgid ) & nvidb_watchdog=$!`，
     `nvidb_finish` 里 `kill $nvidb_watchdog`。节点都是 Linux，`setsid` 保证 pgid 可用；
     macOS `include_local` 走 `nohup` 分支时退化为只 kill `$$`。
   - `executor.probe` 多输出一列 `timeout` 标记，`JobProbe` 加 `timed_out`；
     `_reconcile_jobs` / `_reconcile_staged` 看到 `finished and timed_out` → 状态
     `timeout`（而不是 `failed (exit 143)`），`job_timeout` 事件与告警照旧。

b. **控制端的 `_enforce_timeouts` 改成兜底 + 提醒**：
   - 只对**本 tick 探测成功且 `observed.alive`** 的作业动手，并加 `timeout_grace`
     （默认 600s）——正常情况下节点侧已经先杀掉了。
   - 节点不可达、或作业没被探测到：**不改状态**；超过限时只记一次
     `job_overdue` 警告（新列 `jobs.overdue_at` 去重），文案说明"节点不可达，回来后
     按节点记录结算"。
   - 净效果：`node_down` 期间等于暂停计时，且不需要引入"暂停时长"的簿记。

c. **自愈已经错记的记录**（兜底）：`_reap_orphans` 在 kill 之前先读
   `run_dir` 的 `exit_code`；若存在且 `finished_epoch` 早于记录的 `finished_at`，
   说明作业其实早已结束：改写为 `completed`/`failed`（按退出码）、写回
   `exit_code`/`finished_at`/`result`，发 `job_corrected` 事件；下游的 hold 由 3.3
   自动解除。这条对现有数据库里那 18 条记录也生效（它们的 orphan 行还在）。

**验收**：`tests/test_sched_lifecycle.py` 新增：节点 offline 时超过 `max_runtime_s`
的作业保持 `running` 且只产生一条 `job_overdue`；节点恢复后按 `exit_code` 结算；
runner 子进程测试验证 `timeout` 标记文件。

### 3.3 依赖恢复后自动解 held（P3）

**现状**：`_unhold` 只在 `_gate` 里触发，而 `_gate` 只对自由池全部作业和 lane 窗口内
的作业调用；窗口外的 held 作业（4248）永远停在 held，`release` 又会把依赖删掉。

**改动**

- tick 里 dispatch 之前加 `_refresh_holds()`：遍历 `dbm.held_jobs()`（数量小），
  重算 `_dependencies_ready`；`hold_reason` 变空则 `_unhold`，变了则更新文案。
- `requeue(dep)` 事务内顺手对 `dbm.dependents_of(dep)` 做同样的事，让
  `job requeue` 之后立刻 `job show` 就是准的。
- 自动解除时把对应未确认的 `job_held` 告警一并 `acknowledged`（`kind=job_held
  and job_id=…`），避免告警列表继续催人。
- `release` 保留，语义不变（明确要放弃死依赖时用）。

### 3.4 lane 策略：`strict` / `ready-first`（P4）

**现状**：`_lane_specs` 与 `_dispatch_lanes` 都在第一个未就绪作业处 `break`；
`concurrency>1` 时队头一个等依赖的作业让整条 lane 空转。用户靠几十次 `lane move` 补。

**改动**

- `lanes` 表加 `policy TEXT DEFAULT 'strict'`；CLI
  `nvidb queue lane <lane> policy strict|ready-first`；`lane ls`/TUI 显示。
- 写进 spool 的 `lane.json`（见 3.5），runner 据此选择：
  - `strict`：只看 seq 最小的未 claim spec，不就绪就等（今天的语义）。
  - `ready-first`：取 seq 最小的**就绪**spec；未就绪的原地保留顺序，就绪后仍按
    seq 优先。held（死依赖）在 `ready-first` 下被跳过，在 `strict` 下仍阻塞。
- 控制端 `_dispatch_lanes`（`runner: off` 的路径）同步实现，保证两条路径语义一致。
- 不做"`--after` 软顺序"这类按依赖粒度的开关：lane 级一个策略就够，且不改 `--after` 的含义。

### 3.5 lane 参数热更新与 runner 重启（P5）

**改动**

- spool 里新增 `lane.json`：`{"concurrency", "paused", "policy", "headroom_mb",
  "max_jobs", "digest"}`，由 sync 按 digest 差量写入；runner 每轮循环重读。
  `Runner.concurrency()` 改读它；spec 里的 `concurrency` 字段保留一版作为
  `lane.json` 缺失时的回退，下一版删除。
- `Runner.run` 里 `self.stopping = os.path.exists(STOPFILE)`，删掉 stop 文件即恢复。
- `lane runner restart` 改为直接走"版本替换"路径（`kill -TERM` 旧 runner → 起新
  runner → `adopt` 正在跑的作业），不再经过 stopping 阶段；`--hard` 保留为别名。
- `set_lane_paused` 也走 `lane.json`（现在靠"不暂存 spec"实现，整条 lane 下发后
  需要显式的 paused 标志）。

### 3.6 按显存而不是数量控制 lane 并发（P6）

**现状**：runner 只比较 `nvidia-smi memory.free` 与队头 `vram_mb`；`concurrency`
是唯一真正的门槛。

**改动**（全部在 runner 内）

- 每轮采集 `nvidia-smi --query-compute-apps=pid,used_memory` + `ps -eo pid,pgid`，
  按 pgid 把显存归到 `self.current` 的作业上（与控制端 `_build_gpu_states` 同一套
  归属规则），得到每个作业的 `measured_mb`。
- 预留记账：`charge_j = max(reserved_j, measured_j)`；
  `free_effective(gpu) = free_now − Σ_j max(0, reserved_j − measured_j) − headroom_mb`。
  刚启动还没申请显存的作业按预留全额计入，正是现在缺的那一步。
- 起下一个的条件：`len(current) < concurrency` **且** 对 lane 每张卡
  `free_effective ≥ vram_next`。无法取得逐进程数据（WSL blind）时 `measured=0`，
  自动退化为保守的"按预留计"。
- `concurrency` 允许 `auto`（= `max_jobs_per_gpu`），推荐 A100 lane 设 `auto`
  让显存说话，round(~20G) 与测试(~17G) 自然混跑。
- `headroom_mb`/`max_jobs` 随 `lane.json` 下发，避免节点与控制端两套阈值。

**验收**：runner 子进程测试用 PATH 上的假 `nvidia-smi`（已有
`test_the_vram_gate_holds_a_job_back_only_when_it_can_measure` 的做法）验证
"4×30G 在 80G 上只起 2 个，第 3 个等到有人退出"。

### 3.7 节点能力标签 + 候选 lane（P7）

**改动**

a. **标签**：`servers[].tags: [vllm, train]`（queue.yml / config.yml）；
   `sync_nodes_from_config` 写进 `nodes.tags`。作业新列 `requires`（JSON 列表），
   `submit --require vllm`（可重复）。`_find_placement` 与 `lane_assign` 只考虑
   标签齐全的节点；`submit --lane` 指向缺标签的节点时直接报错。
   没有任何已配置节点满足标签时**不 fail**，作业保持 pending 并发一次
   `job_unschedulable` 级别的警告——标签是手工维护的，4090 修好环境加上 `vllm`
   标签后作业应当自动开跑。

b. **候选 lane**：`submit --lane "406:0,406:1,polyu-4090:*"` 或
   `--lane any --require vllm`。作业新列 `lane_candidates`（JSON），`lane` 保持
   NULL 直到真正放置。放置用**拉取**而不是推送：runner 在 `state.json` 里上报
   `idle_slots`（并发与显存都还有余量、且本 lane 没有可运行 spec）；控制端在
   `_sync_lane_runners` 里，对 `idle_slots > 0` 的 lane，从候选池取优先级最高、
   名单包含该 lane 的作业，`lane`/`lane_seq` 置尾（CAS：`lane IS NULL`），记
   `lane_assigned (auto)` 事件，同一 tick 内立刻暂存。多条 lane 同时空闲时按
   `placement` 设置（spread/pack）挑节点。因为只有真正空闲的 lane 才拉，不会出现
   某条 lane 囤积候选作业的情况。不带 `--lane` 的自由池 GPU 作业从此等价于
   `--lane any`（候选 = 满足 `--node`/`--require`/显存的所有 lane），自由池的
   `_launch` 路径退役，所有启动都经 runner。

c. TUI/`queue status` 的 `ANY GPU` 池显示候选列表；`lane ls` 显示 `idle_slots`。

### 3.8 keeper 自启动（P8）

**改动**

- `keeper install --launchd`（macOS）：写
  `~/Library/LaunchAgents/com.nvidb.queue-keeper.plist`，`KeepAlive: true`、
  `RunAtLoad: true`、`ProcessType: Background`，直接运行
  `nvidb queue daemon --interval N`（launchd 自己负责重启，不再需要 sh 监督循环），
  环境变量带 `NVIDB_HOME` 与 `PATH`，日志仍写 `queue-keeper.log`。
  `MANAGERS += "launchd"`，`status/stop/restart/logs` 通过 `launchctl` 实现，
  与现有 systemd 分支对称。`install` 不带 `--manager` 时按平台自动选：darwin →
  launchd，有 systemd 用户会话 → systemd，否则 shell。
- **本机 ensure**：`cli._open()` 在本机是队列主机、存在 `queue-keeper.manager`、
  且 `keeper.status()`（已有 10s 缓存）为 down 时，调一次 `keeper ensure`——
  把 `remote:` 客户端已经享有的行为补给本机命令。daemon 自身与 `keeper` 子命令跳过。
- `cmd_daemon` 捕获 `SchedulerLeaseLost`，记日志继续循环，不再退出进程。
- 合盖睡眠：launchd 在睡眠期间不会跑，这里不装 `caffeinate`（只在接电源时有效，
  且对用户不透明）。P1 之后合盖只影响提交与对账，不影响节点执行，接受这个现实并
  在 README §3.9 写明。

### 3.9 写操作不抢调度锁（P9）

**改动**

- `db.py` 新增 `transition_job(conn, job_id, *, expect_states, **fields) -> bool`：
  `BEGIN IMMEDIATE` 内先读行再写，状态不符返回 False。
- 所有改状态的路径改用它：tick 侧 `_launch`（`pending→running`）、
  `_reconcile_jobs`（`running→终态`）、`_reconcile_staged`、`_adopt_runner_state`、
  `_enforce_timeouts`、`_gate` 的 impossible、`_hold`；用户侧 `cancel`、`requeue`、
  `release`、`edit_job`、`lane_move/assign/skip`。
- 去掉 `cancel`/`requeue`/`lane_skip`/`set_node_enabled` 上的 `_operation_lease`；
  `TICK_LOCK` 只用来串行化 tick 本身。逐个 race 的处理：
  - cancel 与 `_launch` 同时：远端进程已起、CAS 失败 → `_launch` 立即
    `executor.terminate` 自己刚起的 pid；不会泄漏。
  - cancel 与 runner 已 claim：现有 `_stop_unwanted` 已覆盖。
  - requeue 与 reconcile 同时：reconcile 的 CAS 期望 `running`，行已是 `pending`
    → 不写，不会把新一轮 attempt 覆盖成旧终态。
  - lane move 与暂存同时：差量同步下下一 tick 自然重写；已被 claim 的作业移动
    只改数据库顺序，无害。
- `set_node_ignored` 保留锁（要和 dispatch 互斥，且极少用）。
- CLI：`job cancel` 补 `--no-tick`；`ArgumentParser.error` 重写，argv 含 `--json`
  时输出 `{"ok": false, "error": …}` 而不是 usage 文本。

## 4. 数据模型与协议变更汇总

**SQLite（`SCHEMA_VERSION` 2 → 3，`init_schema` 走 `ALTER TABLE ADD COLUMN`）**

| 表 | 新列 | 用途 |
|---|---|---|
| `jobs` | `overdue_at TEXT` | 3.2 超时提醒去重 |
| `jobs` | `requires TEXT` | 3.7 能力标签 |
| `jobs` | `lane_candidates TEXT` | 3.7 候选 lane |
| `lanes` | `policy TEXT DEFAULT 'strict'` | 3.4 |
| `nodes` | `tags TEXT` | 3.7 |
| `events.kind` | `job_corrected`、`job_overdue`、`lane_assigned`（auto） | — |
| `alerts.kind` | `job_overdue` (warning) | — |

**spool 协议（节点上 `~/.nvidb/lanes/<slug>/`）**

- `queue/<seq>-<id>.json`：spec 增 `after[]`、`gated`、`digest`。
- `lane.json`：lane 级参数（3.5）。
- `state.json`：`queued[]` 改为 `{job_id, seq, digest}`；增 `idle_slots`、
  `lane_digest`、`finished[].timed_out`。
- `<run_dir>/timeout`：超时标记（3.2）。
- `RUNNER_SCRIPT` 的 hash 变化触发自动滚动，无人工步骤。

**配置（`queue:`）**

- `lane_lookahead: all`（新默认）；`timeout_grace: 600`；
  `servers[].tags`；lane `concurrency` 允许 `auto`。

**CLI**

- `job submit --require TAG`、`--lane a,b,c` / `--lane any`；
- `job cancel --no-tick`；
- `queue lane <lane> policy strict|ready-first`；`lane concurrency auto`；
- `queue keeper install --launchd`；
- `--json` 下 argparse 错误也是 JSON。
- `skills/nvidb-queue/SKILL.md` 同步补充上述用法（该目录被两个 agent 符号链接）。

## 5. 分阶段落地

按阶段 A、B、C 的依赖顺序实施。阶段 A 已作为一组相互关联的恢复修复完成；
后续每个 PR 只含一个编号项，使用 Conventional Commits。

**版本与提交规则（适用于本文全部阶段）**

- 实施期间只递增 bugfix / patch 版本。阶段 A 首次提交为 `1.9.3`，当前补丁为
  `1.9.4`；后续各项按提交顺序使用 `1.9.5` 等，版本变更与对应改动一起提交。
- 阶段 A、B、C 完成时均不单独提升 minor，也不预先指定 `1.11` 或 `1.12`。
- 整份方案完成并最终合并时，再单独提交 minor 版本变更。按当前 `1.9.x`
  基线，该提交为 `chore(release): 1.10.0`。
- 提交类型仍按改动内容选择 `fix`、`feat`、`refactor` 或 `perf`，不由类型自动
  提升 minor。版本号调整本身不触发服务重启、runner 更新或现有队列配置变更。

**阶段 A（补丁版本 1.9.3）—— 修复本次事故涉及的恢复行为**

1. `fix(sched)`: 3.2b/3.2c —— `_enforce_timeouts` 只凭证据；`_reap_orphans` 自愈错记记录。
2. `fix(sched)`: 3.3 —— `_refresh_holds` + requeue 联动 + 自动 ack。
3. `fix(runner)`: 3.5 —— `stopping` 复位；`lane.json` 承载并发/暂停/策略（先只用并发与暂停）。
4. `fix(cli)`: 3.9 的 CLI 部分 —— `cancel --no-tick`、JSON 化 argparse 错误；
   daemon 不因 `SchedulerLeaseLost` 退出。
5. `feat(keeper)`: 3.8 —— launchd 管理器 + 本机 ensure。

阶段 A 包含控制端修复、`lane.json` 兼容扩展和 runner 安全接管；
实际改动范围与验证结果见文首实施更新。

**阶段 B（继续递增 patch）—— runner v2，减少对控制端在线的依赖**

6. `feat(runner)`: 3.1a/3.1b —— 整条 lane 下发 + 依赖门控 + 差量同步。
7. `feat(runner)`: 3.2a —— 节点侧超时（runner 标记文件 + run.sh 看门狗）。
8. `feat(runner)`: 3.4 —— `ready-first` 策略（runner 与 `_dispatch_lanes` 两处）。
9. `feat(runner)`: 3.6 —— 预留记账的显存门槛 + `concurrency: auto`。
10. `feat(sched)`: 3.1d —— cpu lane。

**阶段 C（继续递增 patch）—— 减少手工操作与分配**

11. `refactor(sched)`: 3.9 —— CAS 状态迁移，撤掉写操作的 `_operation_lease`。
12. `perf(sched)`: 3.1c —— `node_sync` 每节点一次往返。
13. `feat(sched)`: 3.7 —— 标签 + 候选 lane。

依赖关系：6 依赖 3（`lane.json`）；8、9、13 依赖 6；11 与 12 相互独立，可并行。

## 6. 验证

按用户约定只做必要验证：每项一组针对性 pytest，加一次真机冒烟。

- 单测扩展点：`tests/fake_cluster.py` 的 `FakeNode.lane_sync` 需支持差量语义、
  `timeout` 标记、`tags`；`tests/test_sched_runner.py` 已有真子进程跑
  `RUNNER_SCRIPT` 的夹具，新增 ready-first / lane.json 热更新 / stop 复位 /
  显存记账（假 `nvidia-smi`）四个用例。
- 真机冒烟（`nvidb-lanes-and-runner` 记忆里明确写过：runner 改动只靠 fake cluster
  抓不到 `$HOME` 引号、窗口漏观测这类问题）：阶段 B 合并前在 406 上跑一条 3 个作业、
  含 `--after` 与 cpu lane 依赖的 lane，中途 `nvidb queue keeper stop` 30 分钟，
  确认 lane 跑完且对账正确。做一次即可。

## 7. 风险与回滚

- runner 协议变更靠 hash 自动滚动，回滚同样自动：回退版本后第一次 sync 会把
  runner 换回旧脚本，正在跑的作业在自己的 session 里不受影响。
- 整条 lane 下发后，节点上的 spool 里会有更多 spec 文件；`lane_lookahead`
  改回整数即可退回旧行为，不需要迁移。
- schema v3 只加列，旧版本打开新库不会报错（多余列被忽略），可以双向切换。
- CAS 改造（阶段 C 第 11 项）是触及面最广的一项，单独 PR、单独发版，观察一周。

## 8. 明确不做的

- 不把队列改成"必须有 daemon"；不把 lane 默认策略改成 `ready-first`。
- 不做跨节点的依赖自治（依赖在别的节点时仍由控制端 `gated` 开关推进）：
  节点之间没有互相可达的保证，引入节点间通信超出本版范围。
- 不做作业目录自动清理（`polyu-A100` 磁盘问题另议）。
