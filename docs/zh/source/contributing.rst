贡献指南
========

随时欢迎缺陷修复与文档改进。

.. important::

   开发新功能前，请先
   `提交 issue <https://github.com/mujocolab/mjlab/issues>`_，
   以便我们讨论它是否符合项目范围。


开发环境搭建
------------

克隆仓库并同步依赖：

.. code-block:: bash

   git clone https://github.com/mujocolab/mjlab.git && cd mjlab
   uv sync

安装 pre-commit 钩子，在每次提交前捕获格式与 lint 问题：

.. code-block:: bash

   uvx pre-commit install


常用命令
--------

``Makefile`` 为最常见的开发任务提供了快捷方式：

.. code-block:: bash

   make format      # Format code and fix lint errors (ruff)
   make type        # Type check (ty + pyright)
   make check       # Format + type check
   make test-fast   # Run tests, excluding slow ones
   make test        # Run the full test suite
   make test-all    # Format + type check + full test suite

也可以单独运行某个测试以加快迭代：

.. code-block:: bash

   uv run pytest tests/test_rewards.py

类型检查（``make type``）是必需的，未通过的 PR 会被阻止合并。


构建文档
--------

在本地构建文档：

.. code-block:: bash

   make docs

HTML 输出写入 ``docs/_build/``。编辑时如需实时刷新：

.. code-block:: bash

   make docs-watch


提交 pull request
-----------------

1. Fork 仓库并创建功能分支。
2. 完成修改。
3. 运行 ``make test-all``，确认格式、类型检查与测试全部通过。
4. 按照 `Keep a Changelog <https://keepachangelog.com/>`_ 的惯例，在
   ``docs/source/changelog.rst`` 的 "Upcoming version" 小节中按相应类别
   （Added / Changed / Fixed）添加条目。
5. 提交 pull request。


使用 Claude Code 开发
---------------------

仓库根目录包含 ``CLAUDE.md`` 文件，定义了
`Claude Code <https://claude.com/claude-code>`_ 的开发约定、风格指南与
常用命令。由于它记录的正是 CI 中强制执行的规则，对人类贡献者同样具有
参考价值。

项目还在 ``.claude/commands/`` 中提供了共享命令，任何安装了 Claude Code
的贡献者都可以用斜杠命令调用它们。

``/update-mjwarp <commit-hash>``
   把 ``mujoco-warp`` 依赖更新到指定 commit。一步完成编辑
   ``pyproject.toml``、运行 ``uv lock`` 并发起 PR。

   .. code-block:: text

      /update-mjwarp e28c6038cdf8a353b4146974e4cf37e74dda809a

``/commit-push-pr``
   暂存当前改动，提交、推送并发起 PR。
