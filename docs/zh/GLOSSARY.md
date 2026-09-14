# mjlab 中文文档术语表与翻译规范

本文件是 `docs/zh/` 中文文档的统一翻译规范。所有翻译工作必须遵守本规范,以保证
术语一致性和 rst 源文件的结构完整性。

## 一、核心术语对照表

| 英文术语 | 中文译名 | 备注 |
|---------|---------|------|
| entity | 实体 | |
| actuator | 执行器 | |
| sensor | 传感器 | |
| scene | 场景 | |
| terrain | 地形 | |
| observation | 观测 | |
| action | 动作 | |
| reward | 奖励 | |
| termination | 终止 | |
| event | 事件 | |
| curriculum | 课程 | curriculum learning 译为"课程学习" |
| domain randomization | 域随机化 | |
| environment | 环境 | |
| manager | 管理器 | manager-based 译为"基于管理器的" |
| simulation | 仿真 | |
| simulator | 仿真器 | |
| robot learning | 机器人学习 | |
| reinforcement learning | 强化学习 | |
| GPU-accelerated | GPU 加速 | |
| zero-copy | 零拷贝 | |
| mesh | 网格 | |
| joint | 关节 | |
| degree of freedom (DOF) | 自由度 | |
| articulation | 铰接体 | 指带关节的复合刚体结构 |
| rigid body | 刚体 | |
| spawn | 生成 | 指在场景中放置物体 |
| raycast | 光线投射 | |
| camera | 相机 | |
| viewer | 查看器 | |
| recorder | 记录器 | |
| metric | 指标 | |
| checkpoint | 检查点 | |
| policy | 策略 | |
| episode | 回合 | |
| substep | 子步 | |
| distributed training | 分布式训练 | |
| motion imitation | 运动模仿 | |
| randomization term | 随机化项 | |
| batch / batching | 批 / 批处理 | |
| buffer | 缓冲区 | |
| pipeline | 流水线 | |
| command | 指令 | 指任务指令(如速度指令),非 shell 命令 |
| fixed camera | 固定相机 | |
| height scanner | 高度扫描器 | |
| ground truth | 真值 | |
| image / pixel | 图像 / 像素 | |

## 二、保留英文不译的内容

以下内容必须保持英文原文,逐字节不变:

1. **代码块与命令**:所有 `.. code-block::` 内内容、shell 命令、脚本
2. **文件路径与 URL**:如 `src/mjlab/entity/entity.py`、下载链接
3. **API 名称**:类名(`EntityCfg`)、函数名、方法名、参数名、模块路径
4. **配置键**:dataclass 字段名、YAML/TOML 键名
5. **品牌与项目名**:MuJoCo、MuJoCo Warp、Isaac Lab、PyTorch、NVIDIA、uv、
   wandb、rsl_rl、viser、trimesh、tensordict
6. **文件格式名**:MJCF、URDF、USD、ONNX、BibTeX
7. **rst 交叉引用目标**:`:doc:`source/motivation``、`:ref:`xxx``、
   `:class:`Entity`` 的参数部分(反引号内目标)不译;反引号外的显示文本若
   为独立说明文字可译
8. **标题锚点标签**:`.. _label-name:` 保持原样,保证引用完整性

## 三、rst 结构规则

1. **章节层级严格一致**:中文文件的章节标题数量、嵌套顺序必须与英文版
   一一对应,不得增删章节
2. **指令与选项逐字节保留**:`.. automodule::`、`.. autoclass::`、
   `:members:`、`:maxdepth:` 等指令行不做任何修改
3. **toctree 条目路径不变**:`docs/zh/index.rst` 的 toctree 引用路径与
   英文版一致(中文树内相对路径相同)
4. **图片与资源引用不变**:`.. figure::`、`:width:`、`:alt:` 参数保持,
   alt 描述文字可译
5. **代码引用标记保留**:`` ``inline code`` `` 标记内的内容不译
6. **列表与表格结构一致**:有序/无序列表项数一致,表格行列对应

## 四、翻译风格

1. **技术文档语体**:客观、简洁、准确,不逐词硬译;长句拆分为符合中文
   表达习惯的短句
2. **标题简洁**:如 "Installation" → "安装"、"Quick Start" → "快速上手"
3. **首次出现标注原文**:关键概念首次出现时可采用"中文 (English)"格式,
   如"铰接体 (articulation)",后文直接用中文
4. **英文占位**:代词 this/it 指代明确化,避免"它的它的"式直译
5. **标点**:中文内容用全角标点;代码、路径、英文专名周围保持半角
6. **数字与单位**:与英文版一致(如 4096 environments、1e-3)

## 五、构建与验证

- 中文构建命令:`make docs-zh`(等价于
  `MJLAB_DOC_LANG=zh uv run --group docs sphinx-build -j auto -c docs docs/zh docs/_build/zh`)
- 双语一键构建:`make docs-bilingual` 或 `bash docs/build_bilingual.sh`
- 每批翻译完成后必须构建并确认:无 error、无新增 warning、章节结构与
  英文一致
