# EXP-20260818：IsaacLab History5 Mask + LongDropout10（50k 预案）

## 基本信息

- 状态：Running；用户已确认预案，2026-08-18 17:32 CST 启动
- 时间：2026-08-18
- 分支：`recovery/temporal-history-20260803`
- 当前 HEAD：`3344222`
- 工作区：不干净；正式运行时必须保留 run 目录中的 `git/mjlab_soccer.diff`

## 正式运行

- 日志：`logs/rsl_rl/g1_velocity_football/2026-08-18_17-32-05_IsaacLab_history5_mask_longdropout10_flat_seed42_from_walk16000_to50k_wandb`
- 守护服务：`mjlab-isaaclab-history5-longdropout10-50k-seed42.service`
- 启动脚本：`scripts/run_isaaclab_history5_longdropout10_from_walk_to50k_seed42.sh`
- 启动确认：walking `model_16000.pt` 成功加载，已进入 `Learning iteration 0/50000`。
- 启动前验证：注册与转移兼容测试 `35 passed`；64 环境 1 iteration 烟雾训练成功。
- 烟雾训练解析：Actor `525 -> 512 -> 256 -> 128 -> 29` MLP；Critic 当前 110 维 + `10×110` TemporalCNN；23 个奖励项中没有 `command_velocity_envelope` 或 `action_acc_l2`。

## 对照运行与当前结论

### IsaacLab History5 Mask Flat

- 路径：`logs/rsl_rl/g1_velocity_football/2026-08-15_22-26-20_IsaacLab_history5_mask_flat_seed42_from_walk16000_to50k_wandb`
- 任务：`Mjlab-Velocity-Football-IsaacLabAligned-Flat-Unitree-G1`
- Actor 是普通 MLP，单输入 `actor=[525]`；105 维完整观测保留 5 帧后展平。
- 每帧 105 维 = 98 维本体/指令 + 2 维球 XY + 4 维球到双脚 XY + 1 维可见 mask。
- 没有逐帧、整回合或中途传感器 dropout。
- Critic 是当前特权观测 + 10 帧特权历史 TemporalCNN。

### B1 A1R0 LongDropout10

- 路径：`logs/rsl_rl/g1_velocity_football/2026-08-14_11-44-01_B1_A1R0_longdropout10_isaac_actor_dr_flat_seed42_from_walk16000_to50k_wandb`
- 任务：`Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-LegacyCurriculum-Flat-Unitree-G1`
- Actor 是 490 维本体/指令历史 MLP + `10×7` 球历史 causal TemporalCNN，不是 History5 的 525 维 MLP。
- episode 组成：85% 正常移动控球、5% 全回合可见站立、10% 在 2--6 s 时开始失明并持续到回合结束。
- 还额外包含 `command_velocity_envelope=-1.0`、`action_acc_l2=-0.1` 和 `standing_mode_per_episode=true`。

### 结论

1. History5 不会“无球模式”的根本原因不是 mask 维度错误，而是训练分布里几乎没有可学的持续无球状态。球复位在可见区内，所有 dropout 均为 0；`mask=0` 主要出现在球已经跑出视野/控制区、即将终止的坏状态中。
2. History5 因此学到的是“有球控制”，而不是“视觉消失后继续安全跟踪指令”。D435 连续多帧未检测到球对该策略属于分布外输入。
3. LongDropout10 提供了明确的无球训练分布，但与 History5 同时存在 Actor 结构、历史长度、速度包络和动作平滑奖励等差异，不能把两条 50k run 的差异单独归因于 LongDropout10。
4. 下表是最后 1000 iteration 训练 rollout 均值，只用于描述，不是严格同配置评估：

| 指标 | History5 | LongDropout10 |
|---|---:|---:|
| mean reward | 33.608 | 34.534 |
| episode length | 984.85 | 965.52 |
| ball out of control | 0.046686 | 0.214340 |
| fell over | 0.067020 | 0.030013 |
| ball velocity tracking | 0.744290 | 0.665240 |
| robot linear velocity tracking | 0.670670 | 0.680210 |
| angular velocity tracking | 0.097450 | 0.575240 |
| mean action acceleration | 0.658220 | 0.444450 |

## 待确认的新实验

### 实验问题

在不改变 History5 Actor 结构、观测噪声、奖励权重和速度课程的前提下，只加入 LongDropout10 所必需的长时视觉失效数据与配套门控，是否能使 525 维 History5 MLP 学会无球时的安全速度跟踪。

### 名称和起点

- task ID：`Mjlab-Velocity-Football-IsaacLabAligned-History5-LongDropout10-Flat-Unitree-G1`
- run name：`IsaacLab_history5_mask_longdropout10_flat_seed42_from_walk16000_to50k_wandb`
- 起点：`logs/rsl_rl/g1_velocity_football_pretrain/2026-07-23_18-17-07/model_16000.pt`
- 从 walking checkpoint 重新训练到 50k，不从任一足球 50k checkpoint 续训。

### Actor 和 Critic：完全保留 History5

- Actor：`MLPModel`，隐藏层 `(512,256,128)`，ELU，开启观测归一化。
- Actor 输入：`5×105=525`维展平单 `actor` group；不新增 `actor_history`，不使用 Actor CNN/RNN。
- Critic：当前 110 维特权观测 + `10×110` 特权历史 TemporalCNN；CNN 通道 `(256,128,64)`，kernel 3，global average pooling。
- 动作：29 维关节位置目标，与 History5 基线相同。

### 视觉观测和随机化：保留 History5

- 可见区：机器人 yaw 坐标系 `x=[0.05,1.00] m`、`y=[-0.70,0.70] m`。
- 球位置噪声：`[-0.05,0.05] m`；球到脚向量噪声：`[-0.10,0.10] m`。
- 球观测延迟：0--2 个策略步。
- 不使用 episode 共享球 bias、额外 frame noise 或普通逐帧 dropout。
- 关节位置观测 `biased=false`，不注入 `encoder_bias` 事件；其他本体噪声与 History5 一致。

### 唯一主要自变量：LongDropout10 包

- `standing_mode_per_episode=true`：5% 零速站立回合在整个 episode 内固定且始终可见。
- 失明仅从其余 95% 非站立 episode 抽样，条件概率 `0.10/0.95=0.10526315789473685`，即全体 episode 的 10%。
- 失明起点均匀抽样于 episode 开始后 `2.0--6.0 s`。
- `transition_dropout_until_end_probability=1.0`，触发后持续到 episode 结束。
- YAML 保留 `transition_dropout_duration_range_s=(0.2,0.8)`，但因 until-end 概率为 1，实际不生效。
- 失明时同步将球 XY、球到双脚向量和 visibility mask 置零；指令和本体观测不变。
- `sensor_reward_fade_out_s=0.5`；球速度跟踪和 `ball_front_control` 设 `gate_by_sensor_health=true`。
- `sensor_reward_fade_in_s=0.5` 保留，但本实验没有重新可见分支，训练中不触发。
- `ball_out_of_control.ignore_when_sensor_hidden=true`：只豁免外生失明窗口；主动把球踢出视野不豁免。
- episode 构成：85% 正常移动控球 + 5% 可见站立 + 10% 中途失明到结束，三者互斥。
- play/eval 配置关闭人工 dropout，在 sim2sim 中由 D435 真实 mask 驱动。

### 明确不加的 LongDropout10 附加变量

为保持与 History5 的单主变量对照：

- 不加 `command_velocity_envelope`；
- 不加 `action_acc_l2`；
- 不换成 `10×7` Actor 球历史 TemporalCNN；
- 不改奖励权重、速度课程、球复位分布或物理 DR。

### 奖励、课程和终止

- 主奖励权重：球速度 `+1.0`，机器人线速度 `+1.0`，角速度 `+2.0`，`upright=+1.0`，`pose=+1.0`，`ball_front_control=+0.5`。
- 球相对速度、球相对位置和控制区外项权重继续为 0。
- 其他能耗、动作变化率、姿态、足端、自碰撞和膝部触球项保留 History5 配置。
- 速度初始范围：`vx=[-0.25,1.0]`、`vy=[-0.25,0.25]`、`yaw=[-1,1]`，每 5--6 s 重采样。
- 保留 legacy 速度课程：跟踪阈值 0.7，上限 `vx=[-0.5,2.0]`、`vy=[-0.5,0.5]`，每次扩展 0.1。
- episode 20 s；跌倒阈值 0.8 rad；物理球失控边界与 History5 相同。

### PPO 和训练预算

- seed 42，4096 环境，平地，每环境每 iteration 采样 24 步。
- 物理步长 0.005 s，decimation 4，策略频率 50 Hz。
- PPO：5 epoch，4 mini-batch，`lr=1e-3` adaptive，`gamma=0.99`，`lambda=0.95`，clip 0.2，entropy 0.01，desired KL 0.01，max grad norm 1.0。
- 目标 50000 iterations；checkpoint 每 500 iterations 保存，并保存结束 checkpoint。
- W&B project `mjlab`，`upload_model=false`。

## 拟定命令（当前不执行）

```bash
uv run train \
  Mjlab-Velocity-Football-IsaacLabAligned-History5-LongDropout10-Flat-Unitree-G1 \
  --pretrained-checkpoint logs/rsl_rl/g1_velocity_football_pretrain/2026-07-23_18-17-07/model_16000.pt \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 50000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name IsaacLab_history5_mask_longdropout10_flat_seed42_from_walk16000_to50k_wandb
```

## 确认门

用户确认前，不创建 env config，不注册 task ID，不创建/运行 50k 启动脚本。确认时需明确：

1. 是否同意只加 LongDropout10 包，不加速度包络和 `action_acc_l2`。
2. checkpoint 是否确认每 500 iteration 保存。
