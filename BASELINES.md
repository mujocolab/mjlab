# 足球项目基线

本页是当前唯一有效的基线清单。checkpoint 保存在仓库外，不进入 Git；使用前应核对 SHA-256。

## 1. 坐标 Teacher 基线

- Task ID：`Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-LegacyCurriculum-Flat-Unitree-G1`
- Checkpoint：`/home/ut/football_project/log_old/logs/rsl_rl/g1_velocity_football/2026-08-14_11-44-01_B1_A1R0_longdropout10_isaac_actor_dr_flat_seed42_from_walk16000_to50k_wandb/model_49999.pt`
- SHA-256：`b9eff9ad3bbcc043393dd3f1c259a013878f4e85aed2667fb5633290641c5930`
- 角色：旧 DepthStudent 实际使用的教师，也是当前项目的正式教师基线。
- 策略契约：5 帧本体观测 + 10 帧足球历史，Actor/Teacher MLP 为 `512/256/128`。

## 2. DepthStudent 基线

- Task ID：`Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1`
- Checkpoint：`/home/ut/football_project/log_old/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation/2026-08-21_16-21-47_DepthStudent_MountRangeVisualDR_frozenMLP_teacherRollout_alpha025_noDelay_noLongDrop_pixel5_from_B1Teacher49999_seed42_10k_wandb/model_4000.pt`
- SHA-256：`a3319bc2846d4386ebd38478e08f8f0dbe46d33d7a21fa76a15204026d730ae9`
- 角色：旧 DepthStudent 的冻结 MLP 基线；安装范围 `alpha=0–0.25`。

## 3. 当前 DepthStudent 候选

- Task ID：`Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-ConstrainedMLP-Distillation-Flat-Unitree-G1`
- Checkpoint：`/home/ut/football_project/log_old/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation/2026-08-21_23-28-43_DepthStudent_MountRangeStrongVisualDR_constrainedLatent_lastMLP_mixed030_seed42_resume4000_to10k_wandb/model_9999.pt`
- SHA-256：`34fa57c058005826ea1610eeedf1092c58581242cbb79a52ad29acf99f9d97d5`
- 角色：强安装随机化候选；仅开放最后 MLP 层，Student rollout 最终为 30%。

另保留 `Mjlab-Velocity-Football-Flat-Unitree-G1` 作为最小环境 smoke task，不承担正式教师基线身份。

Klavier SchemeA、消融任务和其复制机器人资产已从活动代码中移除；历史结论仍保留在项目管理归档中。
