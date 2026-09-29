Gymnasium MuJoCo tasks
=====================

These eleven tasks use native ``ManagerBasedRlEnv`` configurations. Each task has
its own environment factory under ``tasks/gym_mujoco/config``. Reusable MDP terms
are split into observations, rewards, resets and terminations. Robot state and
control use mjlab's standard ``EntityCfg``, ``EntityArticulationInfoCfg`` and
``EntityData`` interfaces; there is no direct-environment subclass or custom
physics loop. Gymnasium is needed only for the reference tests.

The shared base wires the scene, simulator, action and episode timeout. Each task
chooses its own reset terms and observation layout. Actor and critic groups start
with independent term configurations, so either group can be changed separately.

Assets and controls
-------------------

The XMLs are copied from Gymnasium 1.3.0 and retained under this task's ``assets``
directory, together with their MIT license. Like the existing Cartpole XML, they
are complete benchmark scenes, including floors, targets or manipulated objects.
They do not belong in the reusable robot asset zoo. Each environment cfg declares
its XML, initial state, simulation settings, decimation and episode limit. Asset
helpers only load/attach MJCF; they do not compile a model to infer task settings.
The registry is the sole task catalog, with ``Mjlab-Gym-...-v5`` IDs shared by
training, evaluation and throughput tools.

``XmlActuatorCfg`` wraps the original XML motors. ``JointEffortActionCfg`` writes
native articulation effort targets, which the actuator layer converts to MuJoCo
controls. XML gearing and limits remain in force. This is effort control, not
position control: joint torque is the motor control multiplied by its gear.
Policy action channels use mjlab's native controlled-joint order. This need not
match the XML actuator order: the articulation maps each target to its motor.
Tests compare by joint name rather than assuming identical channel order.
Reusing an existing Gym policy would require an explicit action permutation.
The native environment exposes an unbounded ``Box`` action space; XML actuator
limits still constrain applied motor controls. An RL library that scales actions
from ``action_space`` bounds needs an adapter with the task's motor limits.

Install and run
---------------

Keep the environment, caches and optional tool downloads inside this repository:

.. code-block:: bash

   cd /path/to/mjlab
   export UV_CACHE_DIR="$PWD/.cache/uv"
   export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
   export XDG_CACHE_HOME="$PWD/.cache"
   export WARP_CACHE_PATH="$PWD/.cache/warp"
   uv sync --extra cpu --extra gym --group dev
   uv run --extra cpu --extra gym list-envs

Use ``--extra cu130`` instead of ``--extra cpu`` on a CUDA-capable NVIDIA machine.
The two torch extras are mutually exclusive. If the repository's pinned Python
version is unavailable, pass a supported interpreter explicitly, e.g.
``--python 3.12``. The local development setup uses ``.tools/uv`` and ``.venv``.

.. code-block:: bash

   uv run --extra cu130 train Mjlab-Gym-Ant-v5
   uv run --extra cu130 play Mjlab-Gym-Ant-v5 --agent zero

Play uses one environment. Registered GPU training defaults are:

.. list-table:: Batched PPO defaults
   :header-rows: 1

   * - Tasks
     - Environments x rollout
     - Minibatches x epochs
     - Iteration budget
   * - Ant, HalfCheetah, both pendulums
     - 4096 x 32
     - 8 x 5
     - 1000
   * - Hopper, Walker2d
     - 2048 x 128
     - 32 x 10
     - 500
   * - Humanoid
     - 2048 x 128
     - 32 x 10
     - 1000
   * - HumanoidStandup
     - 4096 x 32
     - 8 x 5
     - 3000
   * - Swimmer
     - 4096 x 128
     - 4 x 5
     - 300
   * - Reacher
     - 4096 x 128
     - 4 x 5
     - 300
   * - Pusher
     - 4096 x 128
     - 4 x 5
     - 600

The 4096 x 32 profile collects 131072 transitions per iteration, with minibatches of
16384 samples and 40 optimizer steps. The 2048 x 128 profile collects 262144
transitions, with minibatches of 8192 samples and 320 optimizer steps. These are
GPU batch settings, not the CPU reference rollout settings. Budgets are upper
bounds; evaluate saved checkpoints rather than assuming the latest is the best.
One training iteration consists of collection followed by PPO optimization;
the number of optimizer steps is epochs times minibatches. Evaluation episode
counts are separate from the training iteration budget. When resuming, count
completed iterations in each training phase rather than checkpoint filenames.
Reacher, Pusher and Swimmer use 524288 transitions per iteration, minibatches of
131072, and 20 optimizer steps. Reacher/Pusher use gamma 0.98; Swimmer retains
gamma 0.9999.

For another batch size, change the native training flags together:

.. code-block:: bash

   uv run --extra cu130 train Mjlab-Gym-Ant-v5 \
       --env.scene.num-envs 1024 --agent.num-steps-per-env 64 \
       --agent.algorithm.num-mini-batches 8 --agent.algorithm.num-learning-epochs 5 \
       --agent.max-iterations 2000

Here each update contains 65536 samples, each minibatch 8192, and the total budget
is again 131072000 transitions. Environment count alone does not fix rollout
horizon: shorter trajectories change GAE's bootstrapping, even at equal batch size.
The minibatch count stays bounded instead of preserving a tiny CPU minibatch size
and creating thousands of optimizer steps. CLI overrides are explicit; there is
no hidden rebatching or learning-rate scaling. Use a smaller batch on CPU.

Task catalog
------------

All IDs start with ``Mjlab-Gym-``. Actor and critic have separately configured
observation groups with identical information, no privileged critic inputs, no observation noise,
and individually named terms. The order follows Gymnasium where practical.
Floating-base tasks use 6D orientation by default, replacing four quaternion
components with six matrix components. All tasks provide one current observation
frame, matching stock Gymnasium: no frame stacking or observation history is
enabled. In mjlab this is represented by term ``history_length=0`` (no history
buffer), not an empty observation.

.. list-table::
   :header-rows: 1

   * - Task suffix
     - Default observations
     - Actions
     - Episode limit
   * - Ant-v5
     - 107
     - 8
     - 1000
   * - HalfCheetah-v5
     - 17
     - 6
     - 1000
   * - Hopper-v5
     - 11
     - 3
     - 1000
   * - Humanoid-v5
     - 350
     - 17
     - 1000
   * - HumanoidStandup-v5
     - 350
     - 17
     - 1000
   * - InvertedDoublePendulum-v5
     - 9
     - 1
     - 1000
   * - InvertedPendulum-v5
     - 4
     - 1
     - 1000
   * - Pusher-v5
     - 23
     - 7
     - 100
   * - Reacher-v5
     - 10
     - 2
     - 50
   * - Swimmer-v5
     - 8
     - 2
     - 1000
   * - Walker2d-v5
     - 17
     - 6
     - 1000

``CartPole-v1`` is a separate Gymnasium classic-control task, with two discrete
actions and analytical dynamics. It is not one of the eleven MuJoCo tasks.
``InvertedPendulum-v5`` is the MuJoCo cart-and-pole benchmark. mjlab also already
provides ``Mjlab-Cartpole-Balance`` and ``Mjlab-Cartpole-Swingup``, which implement
dm_control tasks; they should not be relabeled as Gymnasium CartPole.

Observation configuration
-------------------------

Each task constructs its observation terms in its own cfg. Shared term factories
provide the standard joint or free-root layout; task cfgs select joints, clipping
and additional terms before creating independent actor and critic groups.
``actor_critic_observations`` also accepts separate critic terms for tasks that
need privileged observations later.

Joint position terms reuse the existing ``joint_pos_rel``; there is no new
absolute-position observation term or configurable observation-reference
parameter. Most selected reference positions are zero. Hopper/Walker root height
is observed as ``q_rootz - 1.25`` instead of Gym's ``q_rootz``. This constant shift
preserves the information while leaving the physical reset posture unchanged.
Reacher's target position is supplied by ``generated_commands``. The choice of
reference is unrelated to planar constraints or bounded versus continuous joints.

Reacher's XML contains the arm and a passive target marker in the same scene
asset. The marker's two slide joints are not robot arm joints. A
``ReacherTargetCommand`` samples world XY uniformly in the radius-0.2 disk at
episode reset, synchronizes the marker through Entity write APIs and exposes
the two values through the native command manager. The target remains fixed
throughout the episode. Arm observations select only ``joint0``/``joint1``.

Pusher has a different goal distribution: stock v5 fixes the goal marker's two
slide coordinates at zero and samples only the pushed object's initial position.
The marker's world position includes its XML body offset. Its observation reads
that body position directly, as Gym does; a random target command would change
the benchmark. A command need not be random in general, but a constant command
adds no new information for this fixed-goal task.

The forward-motion tasks do not track sampled velocity commands. Their reward
increases with forward speed, together with the task's survival reward and
control/contact penalties. They therefore have no locomotion command observation
like the velocity-tracking tasks elsewhere in mjlab.

Planar models and the existing Entity API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plane restriction is encoded by the joint tree, not by taking a free root
and adding equality constraints. Hopper, Walker2d and HalfCheetah have root
``slide(x), slide(z), hinge(y)`` joints. Swimmer has
``slide(x), slide(y), hinge(z)``. The other root degrees of freedom simply do not
exist; this is independent of whether an individual hinge has position limits.

``Entity._identify_joints`` removes only a free joint from the articulation's
joint list. A planar model has no free joint, so its root slides/hinges remain in
``data.joint_pos`` and ``data.joint_vel``, including the unactuated coordinates.
The existing fixed-base mocap wrapper positions the whole model; it does not
lock these internal joints or add dynamic degrees of freedom. Consequently its
root pose is not a substitute for the planar torso's generalized coordinates.
No change to Entity, articulation or EntityData is needed for these models.

Selection and reference values belong to different configurations:

* ``SceneEntityCfg`` on each observation term selects joints. Omitting names/IDs
  selects all joints. Selection does not change the articulation or its motors.
* ``EntityCfg.init_state.joint_pos`` supplies the reset/reference posture used by
  ``joint_pos_rel``. Changing it changes the reset posture too; this adaptation
  does not introduce a separate observation offset into Entity or articulation.
* XML ``ref`` belongs to joint kinematics; an action offset belongs to the
  control signal. A position reference must not be put into an effort-action
  offset, which would add a constant force/torque.

The following table states the policy-input semantics. Coordinate selections
follow Gym defaults; quaternion-to-6D conversion and Hopper/Walker's constant
height shift are the explicit representation changes. Horizontal translation is
excluded from position observations by the corresponding Gym default, not by an
Entity limitation. The task cfg selects the remaining joints with
``SceneEntityCfg`` on the existing ``joint_pos_rel`` term; velocities still include
the horizontal root velocity.

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Task
     - Joint and root observations
     - Additional processing
   * - Ant
     - Root height, world orientation (6D), all hinge positions; world root
       linear velocity, body root angular velocity, all hinge velocities.
     - Contact wrenches clipped to [-1, 1]. Horizontal root position excluded.
   * - Humanoid / HumanoidStandup
     - Same root/joint frame conventions as Ant.
     - COM-based inertia/spatial velocity, actuator generalized forces and
       external contact wrenches. Horizontal root position excluded.
   * - HalfCheetah
     - Relative joint positions with zero reference, except ``rootx``;
       all joint velocities.
     - ``rootz`` is a slide coordinate, not a separately derived torso height.
   * - Hopper / Walker2d
     - Relative joint positions except ``rootx``; all joint velocities.
     - ``rootz`` observation is ``q - 1.25``; clip velocities to [-10, 10].
   * - Swimmer
     - Relative joint positions with zero reference, except ``slider1/slider2``;
       all joint velocities.
     - Planar heading is the hinge angle; no free-root quaternion.
   * - InvertedPendulum
     - All relative joint positions (zero reference) and velocities;
       slider and pole hinge.
     - No extra root observation or coordinate selection.
   * - InvertedDoublePendulum
     - Slider position; sin/cos of the two hinges; all joint velocities.
     - Clipped velocity and slider constraint force.
   * - Reacher
     - Arm hinge sin/cos and velocities; command-manager target world XY.
     - Fingertip minus target position in world XY; target velocities excluded.
   * - Pusher
     - Seven relative arm positions (zero reference) and velocities.
     - Tip/object/goal world positions; object/goal slide coordinates are not
       appended again as arm joint observations.

Position references are independent of effort actions. Hopper and Walker2d have
an unactuated root slide joint ``rootz`` with XML ``ref="1.25"``. Its initial
generalized coordinate is 1.25, while the leg hinges have zero reference positions.
Gym observes the root coordinate directly; mjlab subtracts the default pose so
the reset height is observed as zero. Existing policies trained with absolute
height need their observation-normalization mean shifted consistently before
reuse. Reacher's unactuated target slides have ``ref=".1"`` and ``ref="-.1"``;
the command manager overwrites their coordinates with the episode target. They
are marker coordinates, not additional arm joints, action offsets or PD targets.

MuJoCo's XML ``ref`` defines the reference coordinate for joint kinematics;
mjlab's ``default_joint_pos`` is the configured reset posture. They happen to be
initialized consistently here, but are separate concepts. ``joint_pos_rel``
subtracts the latter, not an assumed universal zero. Motor actions remain effort
commands, with no position target and no added action offset.

Action channels follow the native actuator/action configuration; the policy only
requires a consistent channel-to-motor mapping. They need not match Gym's motor
enumeration or the order of observation joints. ``preserve_order=True`` is used
for explicitly ordered body pairs such as fingertip minus target, where swapping
the pair changes the sign of the observation. It does not impose a global joint
ordering requirement on the policy.

The main benchmark-specific requirements are:

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Area
     - Gymnasium convention
     - Adaptation
   * - Root representation
     - Ant/humanoids have free roots; planar robots use unactuated slide/hinge joints.
     - Use root APIs for free roots and articulation joint APIs for planar roots.
   * - Position observations
     - Selected absolute generalized coordinates, often excluding horizontal translation.
     - Existing relative joint-position term; cfg selects coordinates. Hopper/Walker
       root height differs by the constant reset height 1.25.
   * - Orientation and velocity
     - Free-root quaternion, world linear velocity, body angular velocity;
       Reacher/double pendulum use sine and cosine of hinge angles.
     - Replace only existing quaternion observations with 6D; reuse base angular velocity.
   * - Extra dynamics observations
     - Ant contact wrenches; humanoid ``cinert``, ``cvel``, actuator forces and
       contact wrenches; double pendulum constraint force.
     - Small task-local terms with entity indexing; no extra EntityData properties.
   * - Targets and resets
     - Reacher samples a disk; Pusher samples an object position with exclusion;
       some joint velocities use Gaussian noise.
     - Reacher target command; Pusher object reset; native resets wherever applicable,
       with a task-local Gaussian-velocity reset.
   * - Motor control
     - Continuous XML motor controls, gear ratios and limits;
       control cost uses squared actions, not squared geared joint torques.
     - Native XML actuators and effort action terms; fixed channel-to-motor mapping.
   * - Rewards and termination
     - Forward/COM motion, survival, control/contact cost, reaching or standing;
       task-specific health conditions and decision-step time limits.
     - Separate weighted reward and termination terms; numerical differences below.
   * - Physics
     - Per-task timestep/frame skip, RK4 or Euler, contact/friction settings;
       Swimmer fluid effects, Pusher zero gravity, double-pendulum horizontal
       gravity component, and HalfCheetah total-mass normalization.
     - Preserve XML settings; humanoid PGS is replaced by Warp-supported Newton.

.. code-block:: python

   from mjlab.tasks.gym_mujoco.config.ant_env_cfg import ant_env_cfg
   from mjlab.envs import ManagerBasedRlEnv

   cfg = ant_env_cfg(num_envs=4096)
   # Six numbers: the first matrix column, followed by the second column.
   cfg.observations["actor"].terms["root_orientation"].params[
       "representation"
   ] = "rotation_6d"
   env = ManagerBasedRlEnv(cfg, device="cuda:0")

Use the common ``mdp.projected_gravity`` term if a task needs only tilt. The 6D
term supports ``quaternion`` for Gym observation comparisons. Change both actor
and critic terms when matching their representations. For quaternion comparisons,
Ant has 105 observations and the humanoids have 348.

Joint positions, joint velocities, root height/orientation, root velocity,
contact wrenches, target positions and relative positions are separate terms.
Clipping is configured on observation terms, e.g. Hopper/Walker joint velocity
and Ant contact observations. There are no last-action or velocity-command
observations implicitly added to the benchmark.

Common relative joint position, relative joint velocity (zero default velocity), base
angular velocity, survival, timeout and uniform reset terms come directly from
``mjlab.envs.mdp``. Reacher also reuses ``generated_commands``. Task-local code
contains only the additional benchmark terms.

Most terms use existing public articulation properties. The few MuJoCo-specific
compatibility quantities (``cinert``, ``cvel``, ``cfrc_ext``, constraint force) are
isolated in task-local observation/reward terms and selected with entity-local
indexing. They do not expand the public ``EntityData`` API. Native XML force
sensors request post-constraint wrench computation through mjlab's sensor path.

MDP and numerical differences
-----------------------------

This is a manager-based adaptation, not an exact Gymnasium replay wrapper:

* Actor and critic terms run at mjlab's normal post-forward observation point.
  Observation-oracle tests compare the same forwarded state, not unrefreshed
  derived buffers from different points in the simulation loop.
  Rewards and terminations run before that forward call: root/body/site derived
  quantities can be one physics substep old. This is one integration substep,
  not a full policy decision or PPO rollout step. Gym root-health tests read current
  qpos instead. Matching health thresholds therefore does not guarantee identical
  termination times along a trajectory. Extra tests compare the health decisions
  at matched, forwarded states on both sides of the height thresholds.
* Floating roots reset through mjlab's existing pose/velocity API, with Euler
  angle perturbations converted to a unit quaternion. Joint reset noise preserves
  the benchmark's uniform/Gaussian choices; targets use task-specific distributions.
  Uniform joint resets reuse ``reset_joints_by_offset`` and therefore clamp to
  mjlab soft joint limits. The Gaussian-velocity reset remains task-local and
  also clamps sampled joint positions to those limits, including when Gym's
  original reset does not. Velocity noise retains its Gaussian distribution.
  Floating-root noise is intentionally not Gym's additive quaternion noise.
  The native floating-root reset samples velocities uniformly; Ant's original
  Gaussian velocity noise is retained for articulation joints, but not its root.
  The existing joint reset event only supports uniform samples and clamps joint
  positions. The task-local Gaussian-velocity event reuses the existing
  ``sample_uniform``/``sample_gaussian`` utilities and native Entity writes;
  observation-noise configuration does not perturb simulator initial state.
* Forward rewards use articulation forward velocity. Humanoid uses mass-weighted
  COM velocity. Gym uses ``(x_after - x_before) / step_dt``, the average velocity
  over one policy step; this implementation samples instantaneous velocity without
  adding another forward call or changing the core manager update order. The
  difference can decrease with shorter policy steps and smooth motion, but impact
  dynamics and cached body quantities prevent assuming numerical equivalence.
  Other reward weights, unhealthy-state thresholds and time limits follow stock v5.
* Rewards are per decision step, without multiplying by dt. Survival rewards use
  the termination manager and time limits are truncations. In particular, the
  forward component is ``weight * velocity``, not ``weight * velocity * step_dt``:
  at matching policy frequency and episode length its cumulative scale matches
  Gym's sum of displacement divided by policy-step duration. Differences in the
  sampled velocity can still affect the total. Standup's training-only scaling
  experiment is documented separately below; reported evaluation returns are raw.
* XML RK4/Euler selection, timestep, frame skip, density, viscosity and total mass
  are retained. Humanoid tasks replace unsupported PGS with Newton.
* Warp uses float32. Matching observation definitions does not imply identical
  trajectories, learning curves or scores.
* Fixed-base entities use mjlab's standard mocap wrapping. Observation body
  selectors exclude that wrapper, preserving the original benchmark body order.
* The interface is mjlab's batched tensor API, including ``actor``/``critic``
  groups, manager logging and auto-reset. ``gymnasium.make`` is not extended.

PPO configuration and provenance
--------------------------------

``config/rl_cfg.py`` contains explicit v5 task parameters, without legacy
environment IDs or configuration inheritance. Initial parameter choices were
informed by the published `RL Zoo PPO settings
<https://github.com/DLR-RM/rl-baselines3-zoo/blob/630883a18f9b77e15dfaefb8a891633d766ef99c/hyperparams/ppo.yml>`_
and translated to ``RslRlOnPolicyRunnerCfg``. The source experiments used older
environment versions; this is provenance, not a runtime dependency or an official
v5 agent standard. Pusher initially used SB3 PPO defaults because that source
had no Pusher entry.

The mapping retains network size/activation, initial log standard deviation,
learning rate, discount, GAE lambda, clipping, entropy/value coefficients, gradient
clipping as starting values. Rollout size, environment count, minibatches, epochs
and training budget are adapted for Warp as described above. Hopper, Humanoid
and Walker2d use native KL-adaptive learning rate and 2048 x 128 rollouts,
with 32 minibatches and 10 epochs. Each iteration collects 262144 transitions;
each minibatch contains 8192 transitions. Swimmer uses entropy coefficient 0.01
and gamma 0.9999 with 4096 x 128 rollouts, four minibatches and five epochs.
This profile reached 341.76 +/- 2.75 on 1000 held-out episodes (seed 40001)
after 276 iterations from scratch, compared with RL Zoo's 281.56 reference.
The selected Hopper and Walker2d results instead use fresh 4096 x 24 training,
five epochs and four minibatches; they do not validate the registered long-rollout
defaults from scratch. InvertedDoublePendulum also uses native
KL-adaptive learning rate; its 4096 x 32 rollout and reference scalar settings
are retained. Reacher uses adaptive learning rate, 4096 x 128 rollouts,
four minibatches, five epochs, gamma 0.98 and entropy 0.001. It reached
-3.468 +/- 1.367 on 1000 held-out episodes (seed 40001) after 176 iterations
from scratch. This remains below the Minari SAC expert mean of -3.281.
Pusher uses adaptive learning rate, gamma 0.98, 4096 x 128 rollouts and four
minibatches. It reached -28.978 +/- 4.069 on 1000 held-out episodes after 551
iterations from scratch; this remains below the Minari SAC expert mean -22.053.
Pusher uses normalized observations, two 256-unit ReLU layers, lambda 1.0,
entropy 0.01 and five epochs. Their action clipping is 1.0 and 2.0 respectively,
matching Gym's control bounds. A separate Pusher experiment trained 1200 iterations
at 4096 x 32 with four minibatches, then 400 with sixteen minibatches, keeping five
epochs. Its final checkpoint scored -28.900 +/- 4.100 on 1000 episodes: only a
small improvement, not evidence that this continuation profile is better from
scratch. Registered Pusher defaults retain the independently evaluated 128-step
profile. Saved checkpoint agent YAML is the authoritative evaluated configuration.

Additional community PPO comparisons are the
`RAPID Reacher teacher <https://github.com/eastha10/RAPID-Policy-Distillation>`_
(-3.5086 +/- 1.2421 across 100 evaluation episodes) and its
`five-seed Pusher result <https://github.com/eastha10/RAPID-Policy-Distillation/blob/main/results/summaries/pusher_exp012_5seed.json>`_
(-26.1933 +/- 0.7880 across training-seed means). Reacher is at a comparable level;
Pusher remains below that PPO reference. These supplement the SAC references,
not replace them. Different evaluation protocols and the different meanings of
the standard deviations preclude claiming statistically significant superiority.
Humanoid now uses gamma 0.98. A continuation with 2048 x 128 rollouts,
ten epochs and 32 minibatches reached 12049 +/- 1821 and 95.8% timeout on
1000 held-out episodes after 115 additional iterations. Its parent had already
completed 5414 iterations; this is a 5529-iteration lineage, not a 115-iteration
from-scratch result. Fresh convergence with this final gamma has not been
established by that continuation.
Registered Ant, HalfCheetah, InvertedPendulum and HumanoidStandup configs retain
fixed schedules. Reported experiments may use explicit overrides; the selected
checkpoints do not establish fresh convergence for every registered combination.

Trainer behavior still differs: RSL-RL observation normalization is not SB3's
``VecNormalize`` observation/reward normalization and clipping; reward normalization
is not enabled here. Network initialization and Adam defaults are those of the
native RSL-RL models. Except for Reacher/Pusher's explicit native action clipping,
policy actions use native motor limits without an extra wrapper. These configs
are documented starting points, not claims
of reproduced benchmark scores.

Standup reward scale
~~~~~~~~~~~~~~~~~~~~

Gym's default Standup reward is ``height / physics_dt - control_cost -
impact_cost + 1``. Here ``physics_dt=0.003``; it uses absolute height, not height
gain or vertical velocity. A 0.5-metre height already contributes about 167 per
decision, so a 1000-decision return around 160000 is unsurprising and does not
prove a fully upright posture.

The physical frequency is approximately 333.33 Hz. Five physics substeps per
action give a 0.015-second policy step (approximately 66.67 Hz); 1000 policy
steps make a 15-second episode. These settings match Gym's default Standup
timestep, frame skip and time limit. Reward comparison must keep all three
fixed and use the same reward units.

The selected Standup experiment set ``--env.scale-rewards-by-dt True``:
the reward manager multiplies the entire reward by ``step_dt=5*0.003=0.015``.
This reduces critic-target magnitudes without modifying the relative term
weights. It is fixed scaling, not running reward normalization. Evaluation uses
the raw Gym reward. Fresh 4096 x 24 training with five epochs and four minibatches
reached 395345 +/- 15591 on 1000 held-out complete episodes after 2501 iterations.
In a separate posture evaluation of that checkpoint, mean torso height over the
last five seconds was 1.225 m; 95% of
episodes maintained height >=1.0 m and torso-up vertical component >=0.8 during
that interval. Video shows single-leg support: this is a height/orientation
diagnostic, not evidence of natural two-foot standing or an official Gym success
criterion.
The XML's straight-leg reference torso height is about 1.202 m, and its anatomical
up direction is local -X. Reward and termination were not modified to enforce
that diagnostic. A timeout or a high return alone does not establish standing.
The registered environment still defaults to raw rewards and its agent uses a
fixed schedule with 4096 x 32, five epochs and eight minibatches. That complete
default combination has not been validated by the selected fresh run. To run the
evaluated training profile through the normal entry point, apply these overrides
(evaluation must restore ``--env.scale-rewards-by-dt False``):

.. code-block:: bash

   uv run train Mjlab-Gym-HumanoidStandup-v5 \
       --env.scene.num-envs 4096 --env.scale-rewards-by-dt True \
       --agent.num-steps-per-env 24 --agent.algorithm.num-mini-batches 4 \
       --agent.algorithm.num-learning-epochs 5 --agent.max-iterations 2501 \
       --agent.algorithm.schedule adaptive --agent.algorithm.entropy-coef 0.01

Visualization
-------------

All tasks use mjlab's native MuJoCo/Viser viewers and offscreen video recorder.
Each env cfg declares its ``ViewerConfig``: moving bodies are tracked for
locomotion and Standup, Reacher uses a top-down fixed camera, and Pusher uses a
fixed camera covering the arm, object and goal. The task does not implement a
second renderer or viewer loop. Standard policy playback is:

.. code-block:: bash

   uv run play Mjlab-Gym-Pusher-v5 --checkpoint-file /path/to/model.pt \
       --viewer viser --num-envs 1
   uv run play Mjlab-Gym-Reacher-v5 --checkpoint-file /path/to/model.pt \
       --viewer native --num-envs 1

Add ``--video True --video-length 300`` to record through the normal play entry
point, or ``--video True`` during training. The viewer remains interactive until
closed; video length does not stop it. A display server is needed for the native
viewer, while Viser is suitable for remote browser access.

PPO configuration lives in ``config/rl_cfg.py`` alongside environment configs.
There is no task-specific ``rl/runner.py`` because the stock
``MjlabOnPolicyRunner`` is sufficient. Production training and visualization use
the stock ``train`` and ``play`` commands.

Validation
----------

.. code-block:: bash

   uv run --extra cpu --extra gym pytest tests/test_gym_mujoco.py

The tests check grouped observations and reward composition against Gymnasium
at matching states, rotation-6D conversion, joint-name-based actuator and torque
mapping, masses, real Warp stepping, timeout resets and partial resets. Reward
comparisons use instantaneous velocity on both sides to isolate the documented
finite-difference convention from mistakes in reward terms or weights.

Performance evaluation should load the saved agent configuration, use
deterministic actions and count complete episodes from reset. Assigning a fixed
episode quota to each parallel world avoids bias toward quickly failing worlds.
Report raw return, episode length, early failures and timeouts without simultaneous
failure. HalfCheetah, Swimmer, Pusher, Reacher and HumanoidStandup time out even
with an untrained policy, so those tasks require return or behavior measurements.
A short training smoke check establishes that the pipeline runs, not convergence.

`RL Zoo's published PPO table <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md>`_
covers Ant, HalfCheetah, Hopper, Swimmer and Walker2d on v3. The remaining tasks
use `Farama-Minari expert evaluations <https://huggingface.co/farama-minari>`_:
SAC for Humanoid, both pendulums, Pusher and Reacher; PPO for HumanoidStandup.
These are cross-implementation performance targets, not claims of reproducing
the reference algorithm. Reference evaluation sample sizes differ, so their
standard deviations are not uncertainty bounds for a new policy evaluation.

.. list-table:: Published reference returns (not mjlab training results)
   :header-rows: 1

   * - Task
     - Source / algorithm / version
     - Mean return
   * - Ant
     - RL Zoo / PPO / v3
     - 1327.158
   * - HalfCheetah
     - RL Zoo / PPO / v3
     - 5819.099
   * - Hopper
     - RL Zoo / PPO / v3
     - 2410.435
   * - Humanoid
     - Farama-Minari / SAC / v5
     - 8127.004
   * - HumanoidStandup
     - Farama-Minari / PPO / v5
     - 129867.182
   * - InvertedDoublePendulum
     - Farama-Minari / SAC / v5
     - 9356.023
   * - InvertedPendulum
     - Farama-Minari / SAC / v5
     - 1000.000
   * - Pusher
     - Farama-Minari / SAC / v5
     - -22.053
   * - Reacher
     - Farama-Minari / SAC / v5
     - -3.281
   * - Swimmer
     - RL Zoo / PPO / v3
     - 281.561
   * - Walker2d
     - RL Zoo / PPO / v3
     - 3478.798

The throughput script excludes initialization/warmup, synchronizes CUDA and
reports environment transitions and physics substeps separately. It includes
manager overhead and episode resets.

An Ant measurement on an RTX PRO 5000 Blackwell (2026-09-21), with zero actions,
100 warmup decisions and 1000 timed Warp decisions, produced:

.. list-table:: Sampling throughput, excluding policy optimization
   :header-rows: 1

   * - Backend
     - Environments
     - Transitions / second
   * - mjlab / Warp GPU
     - 1
     - 143.78
   * - mjlab / Warp GPU
     - 4096
     - 312999.66
   * - Gymnasium / MuJoCo CPU
     - 1
     - 5423.65

The CPU measurement uses 10000 timed decisions. The batched GPU throughput is
57.71 times the single CPU environment throughput; running just one environment
on the GPU is slower. These measurements do not establish a time-to-convergence
speedup. The numerical precision and reset implementations also differ.

Training budgets count collected transitions: ``num_envs * num_steps_per_env``
per PPO update, summed across all training stages when resuming or changing batch
settings. A transition is one policy decision in one environment; physics
substeps equal transitions multiplied by that task's decimation. Reusing a sample
over PPO epochs does not increase the collected-transition budget. It instead
increases optimizer work: ``num_mini_batches * num_learning_epochs`` optimizer
steps per update. Training reports distinguish measured collection and learning
time, full wall time, update count and cumulative collected transitions. The
standalone sampling throughput above is not an end-to-end training rate or an
average convergence time. Warm starts and concurrent GPU jobs must be identified
when comparing measured training times.

Selected-checkpoint evaluations and training costs are summarized in the
:download:`task benchmark report <../../src/mjlab/tasks/gym_mujoco/BENCHMARK.md>`.
The raw run records are under ignored ``logs/``; the benchmark report is part of
the repository and records the evaluation protocol and reference sources.
Each row pairs the evaluated checkpoint with its completed training prefix.
Stage costs describe the final run; lineage costs include all retained parent
stages. Active wall time excludes evaluation pauses but includes initialization,
logging and saves; process wall time includes pauses while the training process
is alive. Post-exit evaluation time is separate. Reported transitions/second use
collection plus learning time, not full process time. These are measured costs
through selected checkpoints, not earliest convergence times or evidence that
every registered default has converged.
