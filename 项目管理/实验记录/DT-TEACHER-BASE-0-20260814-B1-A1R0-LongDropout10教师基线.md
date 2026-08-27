# DT-TEACHER-BASE-0：B1/A1R0 LongDropout10 教师基线

- 日期：2026-08-14
- 状态：Completed / Frozen as current baseline
- Seed / 环境数：42 / 4096
- 最终 checkpoint：`model_49999.pt`
- SHA-256：`b9eff9ad3bbcc043393dd3f1c259a013878f4e85aed2667fb5633290641c5930`
- 代码：commit `3344222125d6becde4968b3a9cf8accb2cffeb5d` + run 内
  `git/mjlab_soccer.diff`

## 产物

本机路径：

`/home/ut/football_project/log_old/logs/rsl_rl/g1_velocity_football/`
`2026-08-14_11-44-01_B1_A1R0_longdropout10_isaac_actor_dr_flat_`
`seed42_from_walk16000_to50k_wandb/`

大型 checkpoint、TensorBoard 事件和导出模型保留在外部存储，不提交 Git。

## 冻结配置

- 标准 MJLab G1；29维关节位置动作。
- Actor 主输入为五帧本体/控制堆叠，共490维。
- 足球输入为10帧×7维，使用 causal/dilated BallCNN，输出64维 latent。
- Actor MLP 为 `512/256/128`。
- 零足球 bias 和逐帧噪声；球位置与双脚向量分别使用0--2步随机延迟，可见 mask
  不延迟。
- LongDropout10；命令速度包络权重 `-1.0`；action acceleration 权重 `-0.1`。

## 结果与证据边界

iteration 49999 的训练 mean reward 为 `33.55`，mean episode length 为 `956.23`；
训练过程最高记录 reward 为 `38.31`。这些是训练分布指标，不是固定场景成功率。

后续实验必须记录 `baseline_id=DT-TEACHER-BASE-0` 和 checkpoint 哈希。结构不同的
Klavier 策略不能直接作为该教师的续训 checkpoint；DepthStudent 必须在相同教师与统一
评估 manifest 下比较。
