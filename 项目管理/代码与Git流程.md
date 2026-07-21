# 代码与 Git 流程

## 分支策略

- `main`：可复现的稳定状态，不直接进行实验性开发。
- `feat/<topic>`：功能、任务或训练方法。
- `exp/<topic>`：实验性修改；有保留价值时整理后合并。
- `fix/<topic>`：可明确验证的缺陷修复。
- `docs/<topic>`：项目文档和管理记录。

分支不代替实验记录。进行对比的每个实验必须指向一个确定的 commit。

## 每日本地提交

1. 查看 `git status --short` 和 `git diff --stat`，排除密钥、数据集、检查点和视频。
2. 补全当日记录，更新「接手指针」的 commit、状态和唯一下一步。
3. 按改动风险运行定向测试；每日最后一次提交前运行 `make check`。
4. 用 `git diff --cached` 审阅暂存内容，一次提交只表达一个完整意图。
5. 每个工作日结束时至少创建一个本地提交。若当天无有效变更，只写每日记录，
   不创建空提交。

## 质量门禁

遵循仓库 `AGENTS.md`，命令统一使用 `uv run`。

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check
uv run pyright
uv run pytest tests/<test_file>.py
```

- 本地提交前：定向测试 + `make check`。
- 重大更新推送前：`make test`。
- 用户可见变更：在 `docs/source/changelog.rst` 的 Upcoming version 下增加记录。
- 实验必须保存实际命令、commit SHA、配置、seed 和核心指标。

## 提交文本

遵循本仓库历史，使用简洁的英文祈使句标题，说明可观察的改动：

```text
<imperative summary>

<why the change was needed and any non-obvious tradeoff>
```

例如 `Add soccer experiment record templates`。若关联 issue，在正文末尾使用
`Fixes #<number>`，不写在标题；不硬换行提交正文。

## 重大更新与远程推送

以下任一情况视为重大更新：完成里程碑、改变任务/奖励/观测定义、获得可作基线的
实验结果、修复影响训练正确性的问题，或准备跨机器协作。

推送前必须向项目负责人展示：

- 目标远程、分支和将要推送的 commit 列表。
- `git diff <remote>/<branch>...HEAD --stat` 的变更范围。
- 测试、类型检查和 lint 结果。
- 建议的中文更新摘要和最终提交/PR 文本。
- 是否包含配置变更、不兼容变更、外部产物或已知风险。

只有获得明确确认后才执行 `git push`。默认不强制推送，不覆盖远程历史。

## 推送前摘要模板

```text
推送目标：<remote>/<branch>
提交范围：<base>..<head>
主要变更：<1-3 句>
验证结果：<commands and results>
实验结论：<record link or none>
已知风险：<risks or none>
建议文本：<commit/PR text>
请确认是否推送。
```
