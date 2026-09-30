.. _cloud-training:

云端训练
========

本指南介绍如何用 `SkyPilot <https://skypilot.readthedocs.io/>`_ 在
`Lambda Cloud <https://lambdalabs.com/>`_ 上启动训练任务。SkyPilot 负责
开通 GPU 实例、同步代码、运行任务，结束后自动销毁机器。

``scripts/cloud/`` 下有两个 SkyPilot 任务文件：

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 文件
     - 描述
   * - ``train.yaml``
     - 直接用 uv 安装 mjlab。
   * - ``train-docker.yaml``
     - 从 GHCR 拉取预构建的 Docker 镜像，环境可复现。


前置条件
--------

**1. 安装 SkyPilot**

SkyPilot 是本地 CLI 工具，不是项目依赖。安装：

.. code-block:: bash

   uv tool install "skypilot[lambda]"

**2. Lambda Cloud API 密钥**

在 `Lambda Cloud API keys
<https://cloud.lambda.ai/api-keys/cloud-api>`_ 生成密钥。以你的机器命名
（如 ``kevins-macbook``），便于日后区分。

.. code-block:: bash

   mkdir -p ~/.lambda_cloud && chmod 700 ~/.lambda_cloud
   echo "api_key = <your-api-key>" > ~/.lambda_cloud/lambda_keys
   chmod 600 ~/.lambda_cloud/lambda_keys

**3. 验证配置**

.. code-block:: bash

   sky check lambda

你应该看到 Lambda 列在已启用的云之中。

**4. W&B 凭证** *（可选）*

如果要把日志写到 Weights & Biases，安装 ``wandb`` CLI 并登录：

.. code-block:: bash

   uv tool install wandb
   wandb login

凭证会存进 ``~/.netrc``。SkyPilot 任务文件通过 ``file_mounts`` 把该
文件挂载到远程实例，``wandb`` 即自动完成认证，无需环境变量。


快速上手
--------

在仓库根目录执行：

.. code-block:: bash

   sky launch scripts/cloud/train.yaml \
     --env TASK=Mjlab-Velocity-Flat-Unitree-G1

   # Or with Docker:
   sky launch scripts/cloud/train-docker.yaml \
     --env TASK=Mjlab-Velocity-Flat-Unitree-G1
幕后流程：

1. SkyPilot 找到一台有所需 GPU 的可用 Lambda 实例。
2. 开通实例并通过 rsync 上传你的本地代码。
3. 运行 ``setup`` 步骤（uv 安装或 Docker 拉取）。
4. 运行 ``run`` 步骤（训练）。
5. 空闲 5 分钟后实例自动终止。

.. warning::

   Lambda 实例只能 **启动** 或 **终止**，没有暂停或挂起。不要在实例
   内部运行 ``sudo shutdown``——那会让机器进入告警状态并继续计费。
   终止请始终使用 ``sky down``。


常用操作
--------

**列出可用 GPU**

.. code-block:: bash

   sky show-gpus --infra lambda

**选择其他 GPU**

.. code-block:: bash

   sky launch scripts/cloud/train.yaml --gpus H100:1    # 1x H100
   sky launch scripts/cloud/train.yaml --gpus A100:8    # 8x A100
   sky launch scripts/cloud/train.yaml --gpus A10:1     # 1x A10 (cheaper)

.. note::

   两个任务文件都传了 ``--gpu-ids all``，多 GPU 实例会自动使用
   :ref:`分布式训练 <distributed-training>`。申请多块 GPU 时，考虑按
   比例调低 ``MAX_ITERATIONS``。扩展行为详见
   :ref:`distributed-training`。

**覆盖训练参数**

YAML ``envs`` 块中的每个变量都可以用 ``--env`` 从命令行覆盖：

.. code-block:: bash

   sky launch scripts/cloud/train.yaml \
     --env TASK=Mjlab-Velocity-Flat-Unitree-Go1 \
     --env NUM_ENVS=8192 \
     --env MAX_ITERATIONS=10000
**运行自己的任务**

.. code-block:: bash

   sky launch scripts/cloud/train.yaml \
     --env TASK=Mjlab-Velocity-Flat-Unitree-Go1

查看所有已注册任务：

.. code-block:: bash

   uv run list-envs
   uv run list-envs --keyword Velocity  # filter by keyword


超参数搜索
----------

可以把 `W&B Sweeps <https://docs.wandb.ai/models/sweeps/>`_ 与 SkyPilot
结合，在多 GPU 实例上搜索超参数。sweep 控制器运行在 W&B 服务器上；
实例上的每块 GPU 运行一个独立的 sweep 代理，拉取超参数配置、训练并
上报指标。

示例使用 ``method: random``，每个代理独立采样。贝叶斯搜索同样适合并行
代理：代理完成即回报结果，控制器在轮次之间更新其模型。如果使用贝叶斯，
把 ``run_cap`` 设得足够高，让优化器多跑几轮。

涉及四个文件：

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - 文件
     - 描述
   * - ``sweep.yaml``
     - W&B sweep 配置（参数、搜索方法、指标）。
   * - ``sweep-cluster.yaml``
     - SkyPilot 集群定义（资源、setup，无 run 段）。
   * - ``sweep-agent.yaml``
     - 在一块 GPU 上运行 ``wandb agent`` 的 SkyPilot 任务定义。
   * - ``sweep-launch.sh``
     - 便捷脚本：创建 sweep、开通集群、每 GPU 提交一个代理。

**快速上手**

.. code-block:: bash

   ./scripts/cloud/sweep-launch.sh A100:8   # 8 agents on an 8xA100

该脚本创建 W&B sweep、开通集群并按 GPU 提交代理。每个代理用 sweep
控制器采样的不同超参数组合运行训练。

**手动步骤** （想更精细控制时）：

.. code-block:: bash

   # 1. Create the sweep (returns a SWEEP_ID).
   wandb sweep scripts/cloud/sweep.yaml

   # 2. Provision the cluster (runs setup, no agents yet).
   sky launch scripts/cloud/sweep-cluster.yaml \
     -c mjlab-sweep --gpus A100:8

   # 3. Submit one agent per GPU.
   sky exec mjlab-sweep scripts/cloud/sweep-agent.yaml \
     --gpus A100:1 --env SWEEP_ID=<entity/project/sweep_id> -d

在 W&B 仪表盘上或用 ``sky queue mjlab-sweep`` 监控进度。结束后用
``sky down mjlab-sweep`` 销毁集群。


监控
----

Lambda 分配实例可能需要五分钟或更久。可以开第二个终端盯一下：

.. code-block:: bash

   sky status                               # cluster state (INIT, UP, ...)
   sky logs sky-<cluster-name>              # stream logs in real time
   sky logs sky-<cluster-name> --no-follow  # print current logs and exit
   sky queue sky-<cluster-name>             # job queue for the cluster

.. tip::

   如果集群长时间停在 ``INIT``，多半是这种 GPU 已售罄。用 ``sky down``
   取消并换一种 GPU，或加 ``--retry-until-up`` 让 SkyPilot 持续轮询直到
   有余量。

.. code-block:: bash

   sky down sky-<cluster-name>
   sky launch scripts/cloud/train.yaml --retry-until-up


失败任务上迭代
--------------

任务失败时集群仍在运行（也在计费）。你可以在本地修复问题后直接重新
提交，无需等待新实例：

.. code-block:: bash

   sky exec sky-<cluster-name> scripts/cloud/train.yaml
.. important::

   ``sky exec`` 会 rsync 你的最新代码并只重跑 ``run`` 步骤，
   **不会** 重跑 ``setup``。如果你的修复涉及依赖变更，请重新
   ``sky launch``，或 SSH 进去手动执行 setup 命令。

其他有用命令：

.. code-block:: bash

   sky down sky-<cluster-name>  # terminate the instance immediately
   ssh sky-<cluster-name>       # SSH in (SkyPilot configures this for you)


成本管理
--------

.. warning::

   每次会话结束后务必运行 ``sky status`` 确认没有实例仍在运行。被遗忘
   的实例是意外扣费的最常见来源。一键终止所有实例：``sky down -a``。

- 实例默认空闲 5 分钟后自动终止。可以在 YAML（``idle_minutes``）或
  启动时用 ``--idle-minutes-to-autostop`` 修改。
- YAML 中的 ``down: true`` 设置意味着实例停止时被完全终止而非暂停，
  计费彻底停止。


故障排查
--------

**没有可用实例**

Lambda GPU 经常售罄。可以尝试：

- 用 ``--retry-until-up`` 自动轮询。
- 换一种 GPU：``--gpus A100:1``、``--gpus A10:1`` 等。
- 如果你有其他云（GCP、AWS）的凭证，SkyPilot 可以自动降级使用。
