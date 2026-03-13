.. _tutorial-cartpole:

Cartpole: Building Your First Environment
=========================================

This tutorial walks through building a cartpole swingup task from scratch.
A cart slides along a rail with a pole attached by a hinge. The agent
applies force to the cart to swing the pole up and balance it.

.. raw:: html

   <video style="width:80%; display:block; margin:0 auto;" autoplay loop muted playsinline>
     <source src="../../_static/tutorials/cartpole_swingup.mp4" type="video/mp4">
   </video>
   <p style="text-align:center; color:#666; font-size:0.9em; margin-top:0.5em;">
     A trained agent performing the swingup task.
   </p>

The entire task lives in two files: an XML model and a Python module. We
will build both piece by piece, then snap them together at the end.


The XML model
-------------

Every environment starts with a MuJoCo XML that defines the physical
system. For cartpole that means two bodies, two joints, and one motor:

.. code-block:: xml

    <!-- A cart on a rail with a pole attached by a hinge. -->
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0"
             limited="true" range="-1.8 1.8" damping="5e-4"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" mass="1"/>
      <body name="pole_1" childclass="pole">
        <joint name="hinge_1"/>
        <geom name="pole_1"/>
      </body>
    </body>

    <!-- A motor that pushes the cart along the rail. -->
    <actuator>
      <motor name="slide" joint="slider" gear="10"
             ctrllimited="true" ctrlrange="-1 1"/>
    </actuator>

The motor has gear ratio 10 and control range [-1, 1], so the maximum
force is 10 N. ``ctrllimited`` tells MuJoCo to clamp the control signal
internally, so policy outputs outside this range are safe.

The full XML is at ``src/mjlab/tasks/cartpole/cartpole.xml``.


Building the environment
------------------------

Everything else lives in a single file, ``cartpole_env_cfg.py``. An
mjlab environment is made of small, composable pieces. We will define
each piece, then assemble them into a complete config at the end.

Entity: wrapping the XML
^^^^^^^^^^^^^^^^^^^^^^^^

An entity is mjlab's representation of a simulated object. It loads a
MuJoCo XML, attaches actuators to it, and exposes simulation data (joint
positions, velocities, etc.) as batched PyTorch tensors that your
observation and reward functions read from.

To create one we need three things: a function that loads the XML, an
actuator configuration, and an initial state.

.. code-block:: python

    # Load the XML.
    _CARTPOLE_XML = Path(__file__).parent / "cartpole.xml"

    def _get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_CARTPOLE_XML))

    # Tell mjlab to use the motor defined in the XML as is.
    _CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
        actuators=(XmlMotorActuatorCfg(target_names_expr=("slider",)),),
    )

    # Initial joint states. For swingup the pole starts pointing down
    # (hinge = pi). For balance it starts upright (hinge = 0).
    _BALANCE_INIT = EntityCfg.InitialStateCfg(
        joint_pos={"slider": 0.0, "hinge_1": 0.0},
        joint_vel={".*": 0.0},
    )

    _SWINGUP_INIT = EntityCfg.InitialStateCfg(
        joint_pos={"slider": 0.0, "hinge_1": math.pi},
        joint_vel={".*": 0.0},
    )

Now we can snap these together into an ``EntityCfg``:

.. code-block:: python

    # Bundle the spec loader, actuator, and initial state into one config.
    def _get_cartpole_cfg(swing_up: bool = False) -> EntityCfg:
        return EntityCfg(
            spec_fn=_get_spec,
            articulation=_CARTPOLE_ARTICULATION,
            init_state=_SWINGUP_INIT if swing_up else _BALANCE_INIT,
        )

That is the entity done. Later, we will pass it to the scene so the
environment knows what to simulate.

Observations: what the agent sees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each observation term is a function that reads from the simulation and
returns a tensor. The observation manager concatenates them into a single
vector for the policy. mjlab provides common terms in ``mjlab.envs.mdp``
(joint positions, velocities, etc.), but you can always define your own.

For cartpole we need to ask: what does the agent need to know to act
well? The cartpole has two moving parts (the cart and the pole), so its
physical state is fully described by two positions and two velocities.
This is the minimal set that makes the problem Markov: given these four
values, the future trajectory is completely determined by the equations
of motion for any control input. If you removed any one of them (say
the pole velocity), the agent could not tell whether the pole is falling
or recovering, and no policy could reliably act on that.

In dm_control's terminology, this makes the task *strongly observable*:
the full state can be recovered from a single observation.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Term
     - Dim
     - Description
   * - ``cart_pos``
     - 1
     - Where is the cart on the rail?
   * - ``pole_angle``
     - 2
     - Which way is the pole pointing? (cosine and sine)
   * - ``cart_vel``
     - 1
     - How fast is the cart moving?
   * - ``pole_vel``
     - 1
     - How fast is the pole rotating?

The pole angle is encoded as cosine and sine rather than a raw angle.
MuJoCo's unlimited hinge does not wrap the angle, so as the pole spins
the raw value keeps growing. Cosine and sine give the same output for
the same physical angle regardless of how many rotations have occurred.
This is the one custom observation function:

.. code-block:: python

    def pole_angle_cos_sin(env, asset_cfg) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        # joint_pos has shape [num_envs, num_joints]. Unlike vanilla
        # MuJoCo where data is for a single world, all data in mjlab is
        # batched along the first dimension because many environments
        # run in parallel. Every function you write should accept and
        # return tensors with this leading batch dimension.
        angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
        return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)

To wire these up, we create ``ObservationTermCfg`` entries and group them.
``SceneEntityCfg`` scopes each function to specific joints on the entity:

.. code-block:: python

    # Point at the joints we care about.
    cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
    hinge_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1",))

    # Each term is a function + the params to call it with.
    actor_terms = {
        "cart_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": cart_cfg}),
        "pole_angle": ObservationTermCfg(func=pole_angle_cos_sin, params={"asset_cfg": hinge_cfg}),
        "cart_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": cart_cfg}),
        "pole_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": hinge_cfg}),
    }

    # The RL algorithm expects both an "actor" and "critic" group.
    # They see the same observations here; when you add noise later,
    # you can give the critic clean observations for better value
    # estimates (asymmetric actor-critic).
    observations = {
        "actor": ObservationGroupCfg(actor_terms),
        "critic": ObservationGroupCfg({**actor_terms}),
    }

Actions: what the agent does
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The agent outputs a single scalar: the force on the cart.
``JointEffortActionCfg`` writes the policy output to the actuator's
effort target. The ``XmlMotorActuator`` passes it to MuJoCo's ``ctrl``
buffer, which clamps it to [-1, 1] and multiplies by the gear ratio:

.. code-block:: python

    actions = {
        "effort": JointEffortActionCfg(
            entity_name="cartpole",
            actuator_names=("slider",),
            scale=1.0,
        ),
    }

The policy samples from an unbounded Gaussian, so it can output values
outside [-1, 1]. MuJoCo handles the clamping, and the log probability is
computed on the unclamped value so gradients stay correct.

Rewards: the training signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each reward term is a function that returns a scalar per environment.
The reward manager computes a weighted sum of all terms each step.

The cartpole reward reproduces dm_control's smooth reward as a single
multiplicative term:

.. math::

   r = \underbrace{\frac{\cos\theta + 1}{2}}_{\text{upright}}
       \times \underbrace{\frac{1 + g(x)}{2}}_{\text{centered}}
       \times \underbrace{\frac{4 + q(u)}{5}}_{\text{small control}}
       \times \underbrace{\frac{1 + g(\dot\theta)}{2}}_{\text{small velocity}}

Each factor is between 0 and 1. The product is high only when all four
conditions hold simultaneously. A weighted sum would let the agent trade
off one factor against another; the product prevents that.

The offsets (1, 1, 4, 1) control how forgiving each factor is. The
upright term can drop all the way to 0 when the pole is down, making it
the dominant signal. The small_control term bottoms out at 4/5 = 0.8
even at maximum force, so the agent is not afraid to push hard when it
needs to (important for swingup). The centered and small_velocity terms
sit in between, bottoming out at 0.5.

.. code-block:: python

    rewards = {
        "smooth_reward": RewardTermCfg(
            func=cartpole_smooth_reward,
            weight=1.0,
            params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg},
        ),
    }

Terminations: when to stop
^^^^^^^^^^^^^^^^^^^^^^^^^^

The dm_control Control Suite formulates its tasks as infinite-horizon:
there are no terminal states, and the discount is always 1. The
1000 step episodes are just a practical evaluation window, not a real
boundary. The agent should behave as if the episode could continue
forever.

We express this with a time limit termination and ``time_out=True``,
which tells the RL algorithm to bootstrap the value function rather
than treating the end of the episode as a true terminal state:

.. code-block:: python

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

Events: resetting the state
^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the start of each episode, reset events randomize joint positions and
velocities around the initial state we defined in the entity:

.. code-block:: python

    events = {
        "reset_slider": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": SceneEntityCfg("cartpole", joint_names=("slider",)),
            },
        ),
        "reset_hinge": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.034, 0.034),
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_1",)),
            },
        ),
    }

The offsets are relative to the entity's initial state. For swingup the
hinge starts at pi, so the noise keeps it near pointing down.

Snapping everything together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ManagerBasedRlEnvCfg`` is where all the pieces come together. The
scene holds the entity, and the config holds everything else:

.. code-block:: python

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"cartpole": _get_cartpole_cfg(swing_up=swing_up)},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",)),
        ),
        decimation=5,
        episode_length_s=50.0,
    )

``decimation=5`` means the physics runs five substeps per policy step,
giving a 20 Hz control frequency. ``disableflags=("contact",)`` skips
contact computation since cartpole has no collisions. ``num_envs=1`` is
the default; override it from the CLI with ``--num-envs``.


Registration and training
-------------------------

The last step is to register the task so it can be launched by name.
Each registration pairs an environment config with an RL config that
specifies the network architecture and PPO hyperparameters. For cartpole
a small network of two 64 unit hidden layers is plenty. The full RL
config is in ``cartpole_env_cfg.py`` alongside the environment config.

This goes in ``__init__.py``:

.. code-block:: python

    register_mjlab_task(
        task_id="Mjlab-Cartpole-Swingup",
        env_cfg=cartpole_swingup_env_cfg(),
        play_env_cfg=cartpole_swingup_env_cfg(play=True),
        rl_cfg=cartpole_ppo_runner_cfg(),
    )

Train:

.. code-block:: bash

    uv run train Mjlab-Cartpole-Swingup --num-envs 4096

Play back a trained checkpoint, either from a local file or a W&B run:

.. code-block:: bash

    uv run play Mjlab-Cartpole-Swingup --checkpoint-file logs/rsl_rl/cartpole/model_500.pt
    uv run play Mjlab-Cartpole-Swingup --wandb-run-path <user/project/run_id>

.. figure:: ../_static/tutorials/cartpole_training_curve.png
   :width: 70%
   :align: center
   :alt: Cartpole swingup training curve

   Mean reward over 5 seeds (shaded: one standard deviation).

Config fields can be overridden from the CLI:

.. code-block:: bash

    uv run train Mjlab-Cartpole-Swingup \
        --num-envs 8192 \
        --agent.algorithm.learning-rate 3e-4 \
        --agent.algorithm.entropy-coef 0.005


Next steps
----------

**Add observation noise.** The current config has no noise, so the
policy is brittle. Add noise to any observation term to train a more
robust policy:

.. code-block:: python

    from mjlab.utils.noise import UniformNoiseCfg

    ObservationTermCfg(
        func=joint_pos_rel,
        params={"asset_cfg": cart_cfg},
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )

**Randomize the physics.** Use the :ref:`domain_randomization` system to
vary pole mass or joint damping across environments, training a policy
that transfers across physical variations.

**Explore other tasks.** The library ships with locomotion, manipulation,
and motion tracking tasks that you can run out of the box:
``Mjlab-Velocity-Flat-Unitree-Go1``, ``Mjlab-Lift-Cube-Yam``, and
``Mjlab-Tracking-Flat-Unitree-G1``, among others. Reading their source
is a good way to see how more complex reward and observation structures
are composed.

**Build something new.** The cartpole is intentionally minimal. Once you
are comfortable with the pieces, try designing your own robot model and
task from scratch. The same pattern of XML model, observation terms,
reward terms, and config applies regardless of how complex the system
becomes.
