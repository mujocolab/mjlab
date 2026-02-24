.. _entity_data:

Entity Data
===========

This page is the property reference for ``EntityData``. For an overview
of how ``entity.data`` fits into the broader data access story, see
:ref:`entity`.

All properties are PyTorch tensors backed by MuJoCo Warp's GPU buffers
with no copy overhead. The first dimension is always ``num_envs``, the
number of parallel simulation worlds.

.. warning::

   Read properties reflect the state after ``sim.forward()`` is called.
   If you write simulation state and then read a derived property in the
   same event term, call ``sim.forward()`` between the write and the
   read. The environment step sequence already does this; the warning
   applies only when writing custom event terms that mix reads and
   writes. See the :ref:`FAQ <faq-sim-forward>` for a detailed
   explanation.


Reference: root state
---------------------

Root properties describe the position, orientation, and velocity of the
entity's root body. Properties ending in ``_w`` are expressed in the world
frame. Properties ending in ``_b`` are expressed in the entity's base frame.
See :ref:`frame-conventions` for details.

Each entity has two root reference points: the **link origin** (the body
frame origin defined in the MJCF) and the **center of mass (COM)**.
Which one is relevant depends on the task.

.. admonition:: MuJoCo's mixed-frame ``qvel``

   For floating-base entities, the free joint stores 6 DOFs in
   ``qvel``. MuJoCo expresses the **linear** components in the
   **world frame** but the **angular** components in the **local body
   frame**. EntityData avoids this pitfall: all ``_w`` velocity
   properties are computed from ``cvel`` (see
   :ref:`cvel-section` below) and are fully world-frame. If you
   read ``env.sim.data.qvel`` directly, be aware of the mixed
   convention.

.. rubric:: Root link properties

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 30

   * - Property
     - Shape
     - Frame
     - Description
   * - ``root_link_pose_w``
     - ``[num_envs, 7]``
     - world
     - Root link position (3) and quaternion (4) concatenated
   * - ``root_link_pos_w``
     - ``[num_envs, 3]``
     - world
     - Root link position
   * - ``root_link_quat_w``
     - ``[num_envs, 4]``
     - world
     - Root link orientation as quaternion (w, x, y, z)
   * - ``root_link_vel_w``
     - ``[num_envs, 6]``
     - world
     - Root link linear (3) and angular (3) velocity concatenated
   * - ``root_link_lin_vel_w``
     - ``[num_envs, 3]``
     - world
     - Root link linear velocity
   * - ``root_link_ang_vel_w``
     - ``[num_envs, 3]``
     - world
     - Root link angular velocity
   * - ``root_link_lin_vel_b``
     - ``[num_envs, 3]``
     - body
     - Root link linear velocity in base frame
   * - ``root_link_ang_vel_b``
     - ``[num_envs, 3]``
     - body
     - Root link angular velocity in base frame

.. rubric:: Root COM properties

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 30

   * - Property
     - Shape
     - Frame
     - Description
   * - ``root_com_pose_w``
     - ``[num_envs, 7]``
     - world
     - Root COM position (3) and quaternion (4) concatenated
   * - ``root_com_pos_w``
     - ``[num_envs, 3]``
     - world
     - Root COM position
   * - ``root_com_quat_w``
     - ``[num_envs, 4]``
     - world
     - Root COM orientation as quaternion (w, x, y, z)
   * - ``root_com_vel_w``
     - ``[num_envs, 6]``
     - world
     - Root COM linear (3) and angular (3) velocity concatenated
   * - ``root_com_lin_vel_w``
     - ``[num_envs, 3]``
     - world
     - Root COM linear velocity
   * - ``root_com_ang_vel_w``
     - ``[num_envs, 3]``
     - world
     - Root COM angular velocity
   * - ``root_com_lin_vel_b``
     - ``[num_envs, 3]``
     - body
     - Root COM linear velocity in base frame
   * - ``root_com_ang_vel_b``
     - ``[num_envs, 3]``
     - body
     - Root COM angular velocity in base frame

.. rubric:: Derived root properties

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 30

   * - Property
     - Shape
     - Frame
     - Description
   * - ``projected_gravity_b``
     - ``[num_envs, 3]``
     - body
     - Gravity vector (0, 0, -1) rotated into the base frame. Used to measure
       tilt: a perfectly upright robot reads ``[0, 0, -1]``.
   * - ``heading_w``
     - ``[num_envs]``
     - world
     - Heading angle (radians) of the root body's forward axis projected onto
       the XY plane.


Reference: body state
---------------------

Body properties give per-body kinematic state for all bodies belonging to the
entity. The second dimension is ``num_bodies``, which counts all non-world
bodies in the entity's kinematic tree.

.. list-table::
   :header-rows: 1
   :widths: 35 25 15 25

   * - Property
     - Shape
     - Frame
     - Description
   * - ``body_link_pose_w``
     - ``[num_envs, num_bodies, 7]``
     - world
     - Per-body link position (3) and quaternion (4)
   * - ``body_link_pos_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - Per-body link positions
   * - ``body_link_quat_w``
     - ``[num_envs, num_bodies, 4]``
     - world
     - Per-body link orientations
   * - ``body_link_vel_w``
     - ``[num_envs, num_bodies, 6]``
     - world
     - Per-body link linear (3) and angular (3) velocity
   * - ``body_link_lin_vel_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - Per-body link linear velocities
   * - ``body_link_ang_vel_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - Per-body link angular velocities
   * - ``body_com_pose_w``
     - ``[num_envs, num_bodies, 7]``
     - world
     - Per-body COM position (3) and quaternion (4)
   * - ``body_com_pos_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - Per-body COM positions
   * - ``body_com_quat_w``
     - ``[num_envs, num_bodies, 4]``
     - world
     - Per-body COM orientations
   * - ``body_com_vel_w``
     - ``[num_envs, num_bodies, 6]``
     - world
     - Per-body COM linear (3) and angular (3) velocity
   * - ``body_com_lin_vel_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - Per-body COM linear velocities
   * - ``body_com_ang_vel_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - Per-body COM angular velocities
   * - ``body_external_wrench``
     - ``[num_envs, num_bodies, 6]``
     - world
     - External force (3) and torque (3) applied to each body
   * - ``body_external_force``
     - ``[num_envs, num_bodies, 3]``
     - world
     - External forces applied to each body
   * - ``body_external_torque``
     - ``[num_envs, num_bodies, 3]``
     - world
     - External torques applied to each body


Reference: joint state
----------------------

Joint properties cover 1-DOF revolute and prismatic joints. The free joint
(root floating-base DOF) is excluded; use root state properties for that.

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Property
     - Shape
     - Description
   * - ``joint_pos``
     - ``[num_envs, num_joints]``
     - Joint positions in radians (revolute) or metres (prismatic)
   * - ``joint_pos_biased``
     - ``[num_envs, num_joints]``
     - Joint positions with encoder bias added. Used when simulating
       encoder calibration errors via domain randomization.
   * - ``joint_vel``
     - ``[num_envs, num_joints]``
     - Joint velocities in rad/s or m/s
   * - ``joint_acc``
     - ``[num_envs, num_joints]``
     - Joint accelerations in rad/s² or m/s²
   * - ``actuator_force``
     - ``[num_envs, num_actuators]``
     - Scalar actuation force in actuation space. Use this in reward and
       observation terms where joint torques are needed.
   * - ``generalized_force``
     - ``[num_envs, nv]``
     - Generalized forces applied to the free-joint DOFs


Reference: geom and site state
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Property
     - Shape
     - Description
   * - ``geom_pose_w``
     - ``[num_envs, num_geoms, 7]``
     - Per-geom position (3) and quaternion (4) in world frame
   * - ``geom_pos_w``
     - ``[num_envs, num_geoms, 3]``
     - Per-geom positions in world frame
   * - ``geom_quat_w``
     - ``[num_envs, num_geoms, 4]``
     - Per-geom orientations in world frame
   * - ``geom_vel_w``
     - ``[num_envs, num_geoms, 6]``
     - Per-geom linear (3) and angular (3) velocity in world frame
   * - ``geom_lin_vel_w``
     - ``[num_envs, num_geoms, 3]``
     - Per-geom linear velocities in world frame
   * - ``geom_ang_vel_w``
     - ``[num_envs, num_geoms, 3]``
     - Per-geom angular velocities in world frame
   * - ``site_pose_w``
     - ``[num_envs, num_sites, 7]``
     - Per-site position (3) and quaternion (4) in world frame
   * - ``site_pos_w``
     - ``[num_envs, num_sites, 3]``
     - Per-site positions in world frame
   * - ``site_quat_w``
     - ``[num_envs, num_sites, 4]``
     - Per-site orientations in world frame
   * - ``site_vel_w``
     - ``[num_envs, num_sites, 6]``
     - Per-site linear (3) and angular (3) velocity in world frame
   * - ``site_lin_vel_w``
     - ``[num_envs, num_sites, 3]``
     - Per-site linear velocities in world frame
   * - ``site_ang_vel_w``
     - ``[num_envs, num_sites, 3]``
     - Per-site angular velocities in world frame


Reference: tendon state
-----------------------

Tendon properties are only populated for entities that have tendon-driven
actuators.

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Property
     - Shape
     - Description
   * - ``tendon_len``
     - ``[num_envs, num_tendons]``
     - Tendon lengths
   * - ``tendon_vel``
     - ``[num_envs, num_tendons]``
     - Tendon velocities


.. _frame-conventions:

Frame conventions
-----------------

Property names encode their reference frame with a suffix.

``_w`` (world frame)
    A fixed global frame. The origin is typically at the scene origin and
    its axes are constant throughout the episode. World-frame quantities are
    useful when you need absolute position, such as checking whether the
    robot has fallen below a height threshold.

``_b`` (body frame / base frame)
    The entity's root body frame. It translates and rotates with the robot.
    Most observation terms use body-frame quantities because they are
    invariant to the robot's heading direction. A velocity expressed in the
    body frame reads the same whether the robot faces north or south, which
    makes it easier for the policy to generalize.

``projected_gravity_b`` is a good example of why the frame suffix
matters. It takes the world-frame gravity vector ``[0, 0, -1]`` and
rotates it into the base frame. When the robot is upright the result is
``[0, 0, -1]``; as the robot tilts, the x and y components grow,
giving the policy a direct signal for orientation correction.

Quaternion convention
^^^^^^^^^^^^^^^^^^^^^

All quaternions use the ``(w, x, y, z)`` convention, matching MuJoCo.

Reduced state vs. derived quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EntityData properties fall into two categories that behave differently
with respect to ``sim.forward()``:

**Reduced state.** ``joint_pos`` and ``joint_vel`` read directly from
MuJoCo's ``qpos`` and ``qvel`` arrays. Write methods such as
``write_joint_state_to_sim()`` modify these arrays directly, so reads
are always current.

**Derived quantities.** All pose and velocity properties (``*_pose_w``,
``*_vel_w``, ``*_vel_b``) are computed from MuJoCo's internal arrays
(``xpos``, ``xquat``, ``cvel``, ``subtree_com``, etc.) which are only
updated when ``sim.forward()`` runs. If you write to ``qpos``/``qvel``
and then read a derived property without an intervening ``forward()``,
the read will return stale values.

The environment step sequence calls ``forward()`` at the right time, so
this only matters if you write custom event terms that both write and
read in the same function. See the :ref:`FAQ <faq-sim-forward>` for
details.

.. _cvel-section:

How velocity properties are computed from ``cvel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MuJoCo does not store world-frame linear velocities directly. Instead,
it stores a 6D spatial velocity per body called ``cvel`` (com-based
velocity), laid out as ``(angular[3], linear[3])``. This vector is
expressed in the **c-frame**: a frame centered at ``subtree_com`` (the
center of mass of the body's kinematic subtree) and oriented like the
world frame. MuJoCo uses this representation to improve numerical
precision for mechanisms far from the world origin. See
`c-frame variables <https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#c-frame-variables>`_
and Featherstone's
`Spatial Algebra <http://royfeatherstone.org/spatial/>`_ for background.

To recover the world-frame linear velocity at an arbitrary point
:math:`\mathbf{p}` on a rigid body, we apply the standard rigid-body
velocity transfer formula. Let :math:`\boldsymbol{\omega}` and
:math:`\mathbf{v}_c` denote the angular and linear components of
``cvel``, and let :math:`\mathbf{c}` denote ``subtree_com``. Because
the c-frame is world-aligned, :math:`\boldsymbol{\omega}` is already in
the world frame. The linear velocity at :math:`\mathbf{p}` is:

.. math::

   \mathbf{v}_p
     = \mathbf{v}_c
       - \boldsymbol{\omega} \times (\mathbf{c} - \mathbf{p})

EntityData applies this formula in ``compute_velocity_from_cvel()``:

.. code-block:: python

   def compute_velocity_from_cvel(pos, subtree_com, cvel):
       lin_vel_c = cvel[..., 3:6]
       ang_vel_c = cvel[..., 0:3]
       offset = subtree_com - pos
       lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
       ang_vel_w = ang_vel_c
       return torch.cat([lin_vel_w, ang_vel_w], dim=-1)

Every velocity property in EntityData (``root_link_vel_w``,
``body_link_vel_w``, ``geom_vel_w``, ``site_vel_w``, and their COM
variants) uses this function, substituting the appropriate point:

- **Link velocities** use ``xpos`` (body frame origin).
- **COM velocities** use ``xipos`` (body center of mass).
- **Geom/site velocities** use ``geom_xpos``/``site_xpos``, with
  ``cvel`` looked up from the parent body.


Default pose and relative quantities
--------------------------------------

``entity.data.default_joint_pos`` holds the joint positions from the entity's
initial-state configuration (the ``init_state.joint_pos`` field of
``EntityCfg``). It has shape ``[num_envs, num_joints]`` and is replicated
across all environments at initialization time.

The relative joint position is the deviation of the current joint position
from this default:

.. code-block:: python

    joint_pos_rel = joint_pos - default_joint_pos

This is what the ``joint_pos_rel`` observation function computes:

.. code-block:: python

    def joint_pos_rel(env, asset_cfg):
        asset = env.scene[asset_cfg.name]
        jnt_ids = asset_cfg.joint_ids
        return (
            asset.data.joint_pos[:, jnt_ids]
            - asset.data.default_joint_pos[:, jnt_ids]
        )

Relative joint positions give the policy a compact representation of posture
deviation. When the robot is at its default pose, every element is zero.

Similarly, ``default_joint_vel`` is used by the ``joint_vel_rel`` observation
function. For most configurations the default velocity is zero, so
``joint_vel_rel`` is identical to ``joint_vel``. The indirection exists to
allow non-zero reference velocities in tasks such as motion imitation.

The ``use_default_offset=True`` option in joint position action configs uses
``default_joint_pos`` as the zero point for the action space, so a network
output of zero commands the robot to its default pose. This is the standard
configuration for locomotion tasks.
