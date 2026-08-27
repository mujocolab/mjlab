# 实验记录

所有实验先登记在下表中，每个实验只占一行。大型日志、checkpoint 和视频保留在 Git
之外，表格只记录稳定路径和可复盘结论。

| ID | 日期 | 实验名称 | 状态 | 代码版本 | 相对基线变更 | Log 路径 | 结果 | 结论/下一步 |
|---|---|---|---|---|---|---|---|---|
| E0 | 2026-07-22 | MJLab 足球训练基线 | Completed（训练至 7506 iter） | 分支 `feat/football-environment`；run 基于 `0286b85` 和工作区差异 `git/mjlab_soccer.diff`，后固化为 commit `3de262e` | 无，作为后续实验基线 | `logs/rsl_rl/g1_velocity_football/2026-07-22_14-03-33` | mean reward：-1.98 → 最高 29.63 → 末尾 6.42；episode length：23.56 → 881.17；球失控终止：14 → 0.739；训练约 1.29 h | 已学到基础运动和控球，但奖励峰值后明显回落，尚未稳定收敛。E1 整体同步 Isaac Lab 的速度范围、课程和线速度奖励权重 |
| E1 | 2026-07-23 | Isaac Lab 速度课程同步与真机部署 | Completed（训练至 29999 iter） | 分支 `exp/e1-isaac-curriculum`；commit `2f6ebc8` 加 run 内 `git/mjlab_soccer.diff` | 除速度范围、奖励课程和线速度奖励权重外，run 还包含足球观测改为 yaw 系 XY、0–2 步随机延迟和 520 维 Actor 等工作区变更，因此不是单变量实验 | `logs/rsl_rl/g1_velocity_football/2026-07-23_11-16-03` | 最终 mean reward `42.79`、episode length `966.8/1000`、球失控终止 `0.0417`；真机能跟踪速度并短暂运球，但会用膝盖压球来减速和停球 | [真机部署记录](E1-20260723-真机部署.md)。基础能力成立，但出现不可接受的膝部控球奖励投机；下一步 E2 只允许足部控球 |
| E2 | 2026-07-23 | 仅足部控球约束 | Planned | 基于 E1，版本待创建 | 增加球—膝/小腿等非足部接触监控与惩罚；保留足部触球，具体 body 集合和权重先通过接触日志冻结 | 待训练 | 成功标准：固定减速/停球测试中无膝部压球，仍能完成短暂运球和速度跟踪 | 先记录接触 body 分布，再改奖励；避免在不知道真实接触来源时直接加大惩罚 |
| WALK-0 | 2026-07-31 | 10帧 TemporalCNN 行走预训练 | Completed / Historical | commit `2f6ebc8` + run 内 diff | 10帧×98维全本体历史 TemporalCNN | `logs/rsl_rl/g1_velocity_football_pretrain/2026-07-31_20-15-24_TemporalCNN_Walk_hist10_seed42_15k` | 末100 iter：reward `73.29`、episode length `996.92`、policy std `0.318` | [详细记录](WALK-0-20260731-TemporalCNN行走预训练基线.md)；仅作历史初始化参考，不是下一阶段基线 |
| FB-BASE-0 | 2026-08-08 | B1/A1R0 episode-loss5 envelope30 50k | Completed / Frozen | commit `3344222` + run 内 diff | 10帧7维球轨迹 CNN；5% episode loss；30%速度包络 w=-1 | `logs/rsl_rl/g1_velocity_football/2026-08-08_15-03-15_B1_A1R0_episode_loss5_envelope30_w1_seed42_resume20k_to50k_wandb` | 末100 iter：reward `20.278`、episode length `968.62`、policy std `0.400` | [详细记录](FB-BASE-0-20260808-B1-A1R0-50k基线.md)；下一阶段唯一配置/checkpoint 基线，先补固定评估 |
| T1 | 2026-08-01 | 10帧 TemporalCNN + 视觉掩码 | Completed | 当前恢复工作区，run 内 diff | 当前帧 + 10帧 TemporalCNN，矩形可见区和随机丢帧 | `logs/rsl_rl/g1_velocity_football/2026-08-01_10-51-45_TemporalCNN_Football_hist10_visualmask_seed42_20k` | 20k mean reward `41.97`、episode length `971.74`、policy std `0.448` | 单 run 稳定，但与 E3 并非严格单变量对照；需要固定评估和多 seed |
| AB-Plan | 2026-08-03 | B1足球轨迹CNN × 球中心奖励 2×2 消融 | Planned | 待冻结当前恢复分支 | 固定矩形视觉掩码、无随机丢帧；当前帧MLP/10帧B1 Causal-Dilated CNN × IsaacLab式球奖励/球中心奖励 | [详细计划](../TemporalCNN与球中心奖励消融计划_2026-08-03.md) | 旧 A1R1 为完整105维历史、global avg和5%随机丢帧，只作历史参考；四个 cell 均需重训 | 先完成严格主消融，再决定是否继续历史长度 5/10/20 |
| H-Pilot | 2026-08-01 | 5/10/20帧历史长度 5k pilot | Completed（短程） | 当前恢复工作区 | 1024环境、seed42，仅改变历史长度 | `2026-08-01_22-44-37_*history5*5k`、`2026-08-01_23-25-33_*history10*5k`、`2026-08-02_00-11-39_*history20*5k` | 最后100 iteration mean reward：`23.20 / 11.30 / 14.50` | 5帧暂时领先，但不能据单 seed、5k 得出结论 |
| H-Invalid | 2026-08-02 | 5/10/20帧错误续训 | Invalid（已删除） | 错误续训脚本 | 遗漏1024环境且5/20帧误用10帧 task | 本地日志与 W&B runs 已于 2026-08-03 删除 | 回合长度降至2--3步，policy std升至1.56--1.67 | 不可用于结论；从正确5k checkpoint 重做 |
| H5-LD10 | 2026-08-18 | IsaacLab History5 Mask + LongDropout10 50k | Running（17:32 CST 启动） | `3344222` + run 内工作区 diff | 以 History5 525维 MLP 为基线，只加 85/5/10 互斥episode、2--6s后失明到结束、0.5s球奖励淡出及外生失明终止豁免 | `logs/rsl_rl/g1_velocity_football/2026-08-18_17-32-05_IsaacLab_history5_mask_longdropout10_flat_seed42_from_walk16000_to50k_wandb` | 已进入 `0/50000`；每500 iter保存 | [完整配置与当前结论](EXP-20260818-IsaacLab-History5-LongDropout10-50k预案.md)；守护服务运行中 |
| DT-TEACHER-BASE-0 | 2026-08-14 | B1/A1R0 LongDropout10 坐标教师 | Completed / Frozen as current baseline | commit `3344222` + run 内 diff | 当前基线本体；标准G1、490维本体历史、10×7足球历史、512/256/128 MLP | `log_old/logs/rsl_rl/g1_velocity_football/2026-08-14_11-44-01_B1_A1R0_longdropout10_isaac_actor_dr_flat_seed42_from_walk16000_to50k_wandb` | iteration 49999：reward `33.55`、episode length `956.23` | [详细记录](DT-TEACHER-BASE-0-20260814-B1-A1R0-LongDropout10教师基线.md)；后续 DepthStudent 统一相对该教师比较 |
| DEPTH-UNFROZEN-035 | 2026-08-21 | StrongVisualDR 全量解冻 MLP mixed rollout | Completed / Candidate | 从教师蒸馏 run `model_4000.pt` 续训 | 深度安装范围 alpha 0.35、解冻全部 MLP、student rollout 最终 100% | `log_old/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation/2026-08-21_18-42-04_DepthStudent_MountRangeStrongVisualDR_unfrozenMLP_mixedRollout_alpha035_seed42_resume4000_to10k_wandb` | step 9999：reward `30.04`、episode length `875.84` | 不升级为基线；需统一固定评估 |
| DEPTH-CONSTRAINED-030 | 2026-08-21 | StrongVisualDR constrained latent/last-MLP mixed30 | Completed / Latest depth candidate | 同一 `model_4000.pt` 分叉 | 仅训练 MLP 末层，latent loss 0.1、anchor loss 0.001、student rollout 最终 30% | `log_old/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation/2026-08-21_23-28-43_DepthStudent_MountRangeStrongVisualDR_constrainedLatent_lastMLP_mixed030_seed42_resume4000_to10k_wandb` | step 9999：reward `71.47`、episode length `970.19` | 训练分布优于全量解冻版，但需固定 sim2sim 后再下结论 |

## 填写规则

- ID 按 `E0`、`E1`、`E2` 递增，失败或中止的实验也保留一行。
- “代码版本”必须记录分支和 commit；工作区不干净时还要记录 diff 路径。
- “相对基线变更”只写实际改变的变量，不重复记录共同训练配置。
- “结果”至少写起点、最好值和末值，避免只记录“奖励上不去”等主观描述。
- 对照实验每次只改变一个主要变量；共同训练配置默认与基线一致。
- 需要保存完整命令、硬件信息或多 seed 统计时，再创建独立详细记录并从表格链接。

详细实验仍可使用 [模板](EXP-YYYYMMDD-NN-模板.md)，但不是每次训练的必填项。
