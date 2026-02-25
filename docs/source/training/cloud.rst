.. _cloud-training:

Cloud Training
==============

This guide walks through launching training jobs on
`Lambda Cloud <https://lambdalabs.com/>`_ using
`SkyPilot <https://skypilot.readthedocs.io/>`_. SkyPilot provisions a GPU
instance, syncs your code, runs the job, and tears down the machine when
it finishes.

Two SkyPilot task files live in ``scripts/cloud/``:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - File
     - Description
   * - ``train.yaml``
     - Installs mjlab directly with uv.
   * - ``train-docker.yaml``
     - Pulls the pre-built Docker image from GHCR for a reproducible
       environment.


Prerequisites
-------------

**1. Install SkyPilot**

SkyPilot is a local CLI tool, not a project dependency. Install it with:

.. code-block:: bash

   uv tool install "skypilot[lambda]"

**2. Lambda Cloud API key**

Generate a key at `Lambda Cloud API keys
<https://cloud.lambda.ai/api-keys/cloud-api>`_. Name it after your
machine (e.g. ``kevins-macbook``) so you can tell keys apart later.

.. code-block:: bash

   mkdir -p ~/.lambda_cloud && chmod 700 ~/.lambda_cloud
   echo "api_key = <your-api-key>" > ~/.lambda_cloud/lambda_keys
   chmod 600 ~/.lambda_cloud/lambda_keys

**3. Verify setup**

.. code-block:: bash

   sky check lambda

You should see Lambda listed as an enabled cloud.

**4. W&B credentials** *(optional)*

If you log to Weights & Biases, install the ``wandb`` CLI and log in:

.. code-block:: bash

   uv tool install wandb
   wandb login

This stores your credentials in ``~/.netrc``. The SkyPilot task files
mount this file onto the remote instance via ``file_mounts`` so that
``wandb`` authenticates automatically, no environment variable needed.


Quick start
-----------

From the repo root:

.. code-block:: bash

   sky launch scripts/cloud/train.yaml \
     --env TASK=Mjlab-Velocity-Flat-Unitree-G1

   # Or with Docker:
   sky launch scripts/cloud/train-docker.yaml \
     --env TASK=Mjlab-Velocity-Flat-Unitree-G1
What happens behind the scenes:

1. SkyPilot finds an available Lambda instance with the requested GPU.
2. It provisions the instance and uploads your local code via rsync.
3. The ``setup`` step runs (uv install or Docker pull).
4. The ``run`` step runs (training).
5. After 5 minutes of idle time the instance is terminated automatically.

.. warning::

   Lambda instances can only be **launched** or **terminated**. There is
   no pause or suspend. Do not run ``sudo shutdown`` from inside the
   instance; it will put the machine in an alert state and billing will
   continue. Always use ``sky down`` to terminate.


Common operations
-----------------

**List available GPUs**

.. code-block:: bash

   sky show-gpus --infra lambda

**Choose a different GPU**

.. code-block:: bash

   sky launch scripts/cloud/train.yaml --gpus H100:1    # 1x H100
   sky launch scripts/cloud/train.yaml --gpus A100:8    # 8x A100
   sky launch scripts/cloud/train.yaml --gpus A10:1     # 1x A10 (cheaper)

.. note::

   Both task files pass ``--gpu-ids all``, so multi-GPU instances
   automatically use :ref:`distributed training <distributed-training>`.
   When requesting more than one GPU, consider scaling
   ``MAX_ITERATIONS`` down proportionally. See
   :ref:`distributed-training` for details on scaling behavior.

**Override training parameters**

Every variable in the YAML ``envs`` block can be overridden from the
command line with ``--env``:

.. code-block:: bash

   sky launch scripts/cloud/train.yaml \
     --env TASK=Mjlab-Velocity-Flat-Unitree-Go1 \
     --env NUM_ENVS=8192 \
     --env MAX_ITERATIONS=10000
**Run your own task**

.. code-block:: bash

   sky launch scripts/cloud/train.yaml \
     --env TASK=Mjlab-Velocity-Flat-Unitree-Go1

To see all registered tasks:

.. code-block:: bash

   uv run list_envs
   uv run list_envs --keyword Velocity  # filter by keyword


Hyperparameter sweeps
---------------------

Use `W&B Sweeps <https://docs.wandb.ai/models/sweeps/>`_ with SkyPilot
to search hyperparameters across a multi-GPU instance. The sweep
controller lives on the W&B servers; each GPU on the instance runs an
independent sweep agent that pulls a hyperparameter configuration,
trains, and reports metrics.

The example uses ``method: random``, where each agent samples
independently. Bayesian search also works well with parallel agents.
Agents report results back as they finish and the controller updates its
model between rounds. If using Bayesian, set ``run_cap`` high enough for
the optimizer to go through several rounds.

Four files are involved:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - File
     - Description
   * - ``sweep.yaml``
     - W&B sweep configuration (parameters, search method, metric).
   * - ``sweep-cluster.yaml``
     - SkyPilot cluster definition (resources, setup, no run section).
   * - ``sweep-agent.yaml``
     - SkyPilot job definition that runs ``wandb agent`` on one GPU.
   * - ``sweep-launch.sh``
     - Convenience script that creates the sweep, provisions the
       cluster, and submits one agent per GPU.

**Quick start**

.. code-block:: bash

   ./scripts/cloud/sweep-launch.sh A100:8   # 8 agents on an 8xA100

This creates a W&B sweep, provisions a cluster, and submits one agent
per GPU. Each agent runs training with a different set of
hyperparameters sampled by the sweep controller.

**Manual steps** (if you prefer more control):

.. code-block:: bash

   # 1. Create the sweep (returns a SWEEP_ID).
   wandb sweep scripts/cloud/sweep.yaml

   # 2. Provision the cluster (runs setup, no agents yet).
   sky launch scripts/cloud/sweep-cluster.yaml \
     -c mjlab-sweep --gpus A100:8

   # 3. Submit one agent per GPU.
   sky exec mjlab-sweep scripts/cloud/sweep-agent.yaml \
     --gpus A100:1 --env SWEEP_ID=<entity/project/sweep_id> -d

Monitor progress on the W&B dashboard or with ``sky queue mjlab-sweep``.
When done, tear down the cluster with ``sky down mjlab-sweep``.


Monitoring
----------

Provisioning can take five minutes or more while Lambda allocates the
instance. Open a second terminal to keep an eye on things:

.. code-block:: bash

   sky status                               # cluster state (INIT, UP, ...)
   sky logs sky-<cluster-name>              # stream logs in real time
   sky logs sky-<cluster-name> --no-follow  # print current logs and exit
   sky queue sky-<cluster-name>             # job queue for the cluster

.. tip::

   If the cluster stays in ``INIT`` for a long time, the GPU type is
   likely sold out. Cancel with ``sky down`` and try a different GPU, or
   add ``--retry-until-up`` to let SkyPilot keep polling until capacity
   opens up.

.. code-block:: bash

   sky down sky-<cluster-name>
   sky launch scripts/cloud/train.yaml --retry-until-up


Iterating on a failed job
-------------------------

When a job fails the cluster keeps running (and billing). You can fix
the problem locally and resubmit without waiting for a new instance:

.. code-block:: bash

   sky exec sky-<cluster-name> scripts/cloud/train.yaml
.. important::

   ``sky exec`` rsyncs your latest code and reruns the ``run`` step
   only. It does **not** rerun ``setup``. If your fix involves
   dependency changes, use ``sky launch`` again or SSH in and run the
   setup commands manually.

Other useful commands:

.. code-block:: bash

   sky down sky-<cluster-name>  # terminate the instance immediately
   ssh sky-<cluster-name>       # SSH in (SkyPilot configures this for you)


Cost management
---------------

.. warning::

   Always run ``sky status`` after each session to confirm nothing is
   still running. Forgotten instances are the most common source of
   unexpected charges. To terminate everything at once: ``sky down -a``.

- Instances auto-terminate after 5 minutes of idle time by default.
  You can change this in the YAML (``idle_minutes``) or at launch time
  with ``--idle-minutes-to-autostop``.
- The ``down: true`` setting in the YAML means the instance is fully
  terminated when it stops, not just paused. Billing stops completely.


Troubleshooting
---------------

**No instances available**

Lambda GPUs sell out frequently. A few things to try:

- Use ``--retry-until-up`` to poll automatically.
- Try a different GPU type: ``--gpus A100:1``, ``--gpus A10:1``, etc.
- If you have credentials for other clouds (GCP, AWS), SkyPilot can fall
  back to them automatically.
