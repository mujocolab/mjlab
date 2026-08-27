# Football 命令速查

当前只维护 4 个活动 Task ID、3 个训练脚本。正式 checkpoint 与哈希见 [BASELINES.md](BASELINES.md)。

```bash
cd /home/ut/football_project/mjlab_soccer
```

## 活动任务

| 角色 | Task ID |
|---|---|
| 环境 smoke | `Mjlab-Velocity-Football-Flat-Unitree-G1` |
| 坐标 Teacher 基线 | `Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-LegacyCurriculum-Flat-Unitree-G1` |
| DepthStudent 基线 | `Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1` |
| DepthStudent 候选 | `Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-ConstrainedMLP-Distillation-Flat-Unitree-G1` |

## 训练

重建坐标 Teacher：

```bash
bash scripts/run_longdropout10_isaac_actor_dr_from_walk_to50k_seed42.sh
```

从正式坐标 Teacher 训练冻结 MLP 的 DepthStudent：

```bash
bash scripts/run_depth_student_mount_range_frozen_mlp_teacher_rollout_10k_seed42.sh
```

从 `model_4000.pt` 继续训练强随机化、受约束的 DepthStudent：

```bash
bash scripts/resume_depth_student_mount_range_strong_constrained_model4000_to10k_seed42.sh
```

三个脚本均使用仓库外的 `/home/ut/football_project/log_old/logs` checkpoint，并在启动前检查文件是否存在。

## Sim2sim

坐标策略（足球位置使用仿真真值，并按策略配置完成坐标系、历史和延迟对齐）：

```bash
uv run sim2sim-g1-football --policy /ABS/PATH/TO/policy.onnx
```

深度策略（默认加载当前 constrained candidate 的任务配置）：

```bash
uv run python src/mjlab/scripts/sim2sim/g1_football_depth.py \
  --policy /ABS/PATH/TO/policy.onnx
```

## 检查进程

```bash
ps -ef | rg "Mjlab-Velocity-Football|wandb-core"
```

停止训练时向训练主进程发送 `SIGINT`，使 logger 有机会正常收尾：

```bash
kill -INT PID
```
