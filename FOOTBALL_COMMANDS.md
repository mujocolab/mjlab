# Football 命令速查

```bash
cd /home/ut/football_project/mjlab_soccer
```

## 1. Walk

```bash
uv run train Mjlab-Velocity-Walk-KlavierReplica-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 30001 \
  --agent.save-interval 1000 \
  --agent.upload-model False
```

## 2. 足球 Teacher

```bash
uv run train Mjlab-Velocity-Football-KlavierReplica-BallTemporal-Flat-Unitree-G1 \
  --pretrained-checkpoint /home/ut/football_project/mjlab_soccer/logs/rsl_rl/g1_velocity_walk_klavier_replica/2026-08-24_11-38-37_unitree_g1_flat_copied_model_seed42_30k_wandb/model_20000.pt \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 50001 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name schemeA_zeroBallNoise_syncDelay02_entropy001_symmetry_seed42_50k
```

## 3. 深度 Student 蒸馏

将 Teacher 路径替换为实际 checkpoint：

```bash
uv run train Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-ConstrainedMLP-Distillation-Flat-Unitree-G1 \
  --pretrained-checkpoint /ABS/PATH/TO/TEACHER/model_N.pt \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name DepthStudent_mountStrong_constrained_seed42_10k
```

注意：现有 Student 配置仍是 MLP `512/256/128`，加载本次 `1024/512/256` Teacher 前必须先同步网络尺寸。

## 4. Task 区别

命名含义：

- `A0`：球特征与本体一起进入5帧 MLP。
- `A1`：本体使用5帧 MLP，球的 `10x7` 历史单独进入 BallCNN。
- `R0`：IsaacLab 式球速度奖励，并约束球保持在机器人前方。
- `R1`：球中心/相对位置奖励，不使用 `ball_front_control`。

### Teacher

| Task ID | 特点和用途 |
|---|---|
| `Mjlab-Velocity-Football-KlavierReplica-BallTemporal-Flat-Unitree-G1` | **当前推荐**。Klavier 机器人模型，A1R0，10帧 BallCNN，当前配置为球误差0、7维同步延迟0–2步、长程丢球10%、对称损失系数1.0、推力课程。 |
| `Mjlab-Velocity-Football-A0R0-Flat-Unitree-G1` | 5帧全观测 MLP + R0；用于判断 BallCNN 是否真正有效。 |
| `Mjlab-Velocity-Football-A0R1-Flat-Unitree-G1` | 5帧全观测 MLP + R1；用于单独比较奖励设计。 |
| `Mjlab-Velocity-Football-A1R0-Flat-Unitree-G1` | 5帧本体 MLP + 10帧 BallCNN + R0；旧 B1 Teacher 基础结构。 |
| `Mjlab-Velocity-Football-A1R1-Flat-Unitree-G1` | 5帧本体 MLP + 10帧 BallCNN + R1；同时使用球历史与球中心奖励。 |
| `Mjlab-Velocity-Football-A1R0-History30-Flat-Unitree-G1` | A1R0，但球历史增至30帧；延时更长、网络更深，适合历史长度消融。 |
| `Mjlab-Velocity-Football-A1R0-Dropout10-Flat-Unitree-G1` | A1R0，每帧同步丢球10%；测试短时随机不可见。 |
| `Mjlab-Velocity-Football-A1R0-Dropout5-Flat-Unitree-G1` | A1R0，5%整回合无球；训练真正的无球模式。 |
| `Mjlab-Velocity-Football-A1R0-Dropout5-Envelope30-Flat-Unitree-G1` | 5%整回合无球 + 30%速度包络；限制丢球后的速度偏离。 |
| `Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-LegacyCurriculum-Flat-Unitree-G1` | 10% episode 在2–6秒后持续丢球到结束，带速度包络和旧课程。 |
| `Mjlab-Velocity-Football-A1R0-VisibleOnly-Envelope30-LegacyCurriculum-Flat-Unitree-G1` | 始终可见球；用于得到纯坐标 Teacher，不训练无球行为。 |
| `Mjlab-Velocity-Football-A1R0-VisibilityBlend-Flat-Unitree-G1` | 可见时追球、不可见时按速度指令行走，两种奖励平滑切换。 |
| `Mjlab-Velocity-Football-A1R0-VisibilityBlend-V2-Flat-Unitree-G1` | VisibilityBlend 加分模式、多目标课程，训练更复杂。 |

### 深度 Student

| Task ID | 特点和用途 |
|---|---|
| `Mjlab-Velocity-Football-Depth-TemporalTeacher-Distillation-Flat-Unitree-G1` | 基础10帧深度蒸馏；固定相机，主要验证深度是否能替代球坐标。 |
| `Mjlab-Velocity-Football-Depth-TemporalTeacher-LongDropout10-CameraDR-Distillation-Flat-Unitree-G1` | 相机位置/姿态小范围随机化，并让 Teacher 与深度同步长程丢球。 |
| `Mjlab-Velocity-Football-Depth-TemporalTeacher-DeploymentRobustV2-Distillation-Flat-Unitree-G1` | 外参、FOV、crop、深度 scale/bias、重复帧和延迟随机化；面向部署鲁棒性。 |
| `Mjlab-Velocity-Football-Depth-TemporalTeacher-CalibratedVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1` | 围绕实测外参随机化；无延迟、无重复帧、无长程丢球，冻结坐标 MLP。 |
| `Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1` | 按真实可调铰链的一维安装范围随机化，`alpha=0–0.25`，冻结 MLP。 |
| `Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1` | 更强安装随机化，`alpha=0–0.35`、更大 X/Z/pitch 残差，冻结 MLP。 |
| `Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-ConstrainedMLP-Distillation-Flat-Unitree-G1` | 强安装随机化；仅允许最后 MLP 层适配，Student rollout 最终30%，带 latent/action 约束。 |
| `Mjlab-Velocity-Football-Depth-Student-PPO-Flat-Unitree-G1` | 在蒸馏 checkpoint 基础上用 PPO 强化学习微调 Student。 |

切换任务时只替换 `TASK_ID`：

```bash
uv run train TASK_ID --env.scene.num-envs 4096 --agent.seed 42
```

查看全部 Task 和参数：

```bash
uv run train --help
```

## 5. Sim2sim

Teacher：

```bash
uv run sim2sim-g1-football --policy /ABS/PATH/TO/policy.onnx
```

深度 Student：

```bash
uv run python src/mjlab/scripts/sim2sim/g1_football_depth.py --policy /ABS/PATH/TO/policy.onnx
```

## 6. 查询/停止

```bash
ps -ef | rg "Mjlab-Velocity-Football|wandb-core"
kill -INT 123456
```

将 `123456` 替换为训练主进程 PID。
