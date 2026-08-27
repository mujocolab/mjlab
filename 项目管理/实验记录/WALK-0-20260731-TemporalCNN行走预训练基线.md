# WALK-0：10 帧 TemporalCNN 行走预训练基线

## 基本信息

- 状态：Completed / Historical initialization reference
- 负责人：项目负责人（待补充姓名）
- 时间：2026-07-31 20:15 至 2026-08-01 00:57（Asia/Shanghai）
- 相关里程碑/决策：M1；ADR-0008 行走预训练与足球策略兼容布局
- Git 分支：`exp/e1-isaac-curriculum`
- Git commit：`2f6ebc83d00560a3b6bb294399f6b8b55a6ea3c4`
- 工作区是否干净：否；完整状态和差异保存在 run 内 `git/mjlab_soccer.diff`

## 问题与假设

- 问题：10 帧 TemporalCNN football 策略需要结构匹配且稳定的行走初始化。
- 假设：在无球速度任务上训练全本体历史编码器，可提供稳定步态和时序特征初始化。
- 自变量：10 帧全本体历史 TemporalCNN。
- 基线/对照：本 run 没有同预算、同 seed 的 current-frame 严格对照。
- 成功标准：完成 15k；末段回合长度接近上限；无 NaN 或策略坍塌；产物完整。
- 停止条件：NaN、回合长度持续坍塌或策略标准差持续发散。

## 可复现配置

- Task：`Mjlab-Velocity-Football-Temporal-Pretrain-Flat-Unitree-G1`
- 解析后配置：run 内 `params/agent.yaml`、`params/env.yaml`
- Seed：42
- 环境数：4096
- 训练预算：15000 iteration，24 step/env/iteration
- Episode：20 s，最多 1000 step
- Actor/Critic 当前帧：98 维
- Actor/Critic 历史：10 帧 × 98 维，未展平
- TemporalCNN：通道 `256/128/64`，kernel size 3，ELU，global average pooling
- MLP：`512/256/128`，ELU，观测归一化
- PPO：lr `1e-3` adaptive，gamma `0.99`，lambda `0.95`，entropy `0.01`
- 初始命令：x `[-1, 1]` m/s，y `[-0.5, 0.5]` m/s，yaw `[-0.5, 0.5]` rad/s
- 最终课程：x `[-2, 3]` m/s，y `[-0.5, 0.5]` m/s，yaw `[-0.7, 0.7]` rad/s
- 训练命令：见 `scripts/run_temporal_two_stage_seed42.sh` 的 walk 阶段；以 run 内 YAML
  作为实际解析值的最终依据。

## 运行产物

- Run 目录：`logs/rsl_rl/g1_velocity_football_pretrain/2026-07-31_20-15-24_TemporalCNN_Walk_hist10_seed42_15k`
- 冻结 checkpoint：`model_14999.pt`（10,849,631 bytes）
- 导出 ONNX：`2026-07-31_20-15-24_TemporalCNN_Walk_hist10_seed42_15k.onnx`
- TensorBoard：`events.out.tfevents.1785500136.ut-MS-7E07.14032.0`
- Checkpoint SHA-256：`2a641888e6de662182c66caa46edbf261c70adc2f20afd2e89d547c2d4bba280`
- ONNX SHA-256：`ab07b65ff4baa4a11c58136abb0bbe2e0f7188944388b2221da2e5396fb61dfb`
- Agent YAML SHA-256：`b7b7e68eadf66ecbbce3828a076ff0ec272a2c4b3235fc97672d1f8af052d640`
- Env YAML SHA-256：`4628f4f924b20a2380636e0bdea302bbad54784408b940e165b03b842766a8f0`
- Run diff SHA-256：`31e0fbc5e6ed2f8f1993bbfca611d7c01cfd8d7981202afd02e0274b46ca7ea9`

## 结果

末段值为最后 100 iteration 均值；AUC 按 iteration 做梯形积分并除以训练区间长度。

| 指标 | 起点 | 最后一步 | 末段值 | AUC |
|---|---:|---:|---:|---:|
| Mean reward | -1.183 | 74.504 | 73.294 | 73.033 |
| Episode length | 11.61 | 995.96 | 996.92 | 975.81 |
| Policy std | 0.998 | 0.318 | 0.318 | 0.315 |
| XY velocity error | 0.012 | 0.301 | 0.315 | 0.302 |
| Yaw velocity error | 0.045 | 0.525 | 0.532 | 0.539 |
| Linear tracking reward | 0.007 | 1.737 | 1.718 | 1.654 |
| Angular tracking reward | 0.000 | 1.237 | 1.217 | 1.209 |

训练墙钟时间约 4.69 小时，最后 100 iteration 平均吞吐约 88,637 step/s。

### 定性观察与失败模式

- 训练末段平均回合长度接近 1000 步，没有出现坍塌。
- 当前只有训练日志，没有固定命令集 rollout、视频或实机定性记录。
- 速度误差起点很低是因为初始 episode 极短，不可解释为初始策略跟踪更好。

### 偏差与异常

- 单 seed、单 run，不能估计方差。
- 代码基于旧 commit 加大规模工作区差异；复现必须使用 run 内 diff，而不是当前分支代码。
- 该网络编码 98 维全本体历史，与当前 7 维球轨迹 B1 CNN 不兼容。
- 课程状态不随 Actor-only 迁移；football 阶段从自身课程初值开始。

## 结论

- 是否支持假设：部分支持。训练稳定且产物完整，但缺少随机初始化/当前帧严格对照。
- 可以得出的结论：WALK-0 可作为旧式、结构兼容的 10 帧全观测 TemporalCNN Actor 初始化参考。
- 不能得出的结论：不能证明 TemporalCNN 优于 MLP，不能证明 football 或 sim2real 性能提升。
- 后续动作：仅在复盘旧 TemporalCNN 分支时使用；下一阶段基线改为 `FB-BASE-0`。
