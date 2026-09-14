.. _heterogeneous_worlds:

Heterogeneous Worlds
====================

mjlab can run a single batched simulation in which different parallel
worlds use different mesh assets for the same logical entity. World 0
may simulate a cube, world 1 a sphere, world 2 a bowl. All worlds
share the same compiled scene and the same body and joint structure;
only the meshes and the per-geom attributes that travel with them
(friction, contact bits, mass, density, and a few more) differ across
worlds. Articulated props work too (you can have a hinge or slide
below the variant's root), as long as the joint topology matches
across variants. The feature is exposed through ``VariantEntityCfg``.
The full breakdown of what can and cannot vary across variants is in
the next section.


Quickstart
----------

Say you want some parallel worlds to hold a sphere and others to hold
a cone, with a single shared scene running both at once. Define each
variant as a function that returns an ``MjSpec``, then group them
under one ``VariantEntityCfg``:

.. code-block:: python

    import mujoco

    from mjlab.entity import EntityCfg, VariantEntityCfg


    def make_sphere_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        mesh = spec.add_mesh(name="visual")
        mesh.make_sphere(subdivision=3)
        mesh.scale[:] = (0.05,) * 3
        body = spec.worldbody.add_body(name="prop")
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="visual")
        return spec


    def make_cone_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        mesh = spec.add_mesh(name="visual")
        mesh.make_cone(nedge=16, radius=0.04)
        body = spec.worldbody.add_body(name="prop")
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="visual")
        return spec


    object_cfg = VariantEntityCfg(
        variants={
            "sphere": make_sphere_spec,
            "cone":   make_cone_spec,
        },
        assignment={"cone": 2.0},  # twice as many cones as spheres
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    )

Plug the variant entity into a :ref:`scene` exactly like a regular
``EntityCfg``:

.. code-block:: python

    from mjlab.scene import SceneCfg

    scene_cfg = SceneCfg(
        num_envs=4096,
        entities={"object": object_cfg},
    )

Twice as many worlds will hold a cone as a sphere. Variants not listed
in the ``assignment`` dict default to weight 1.0; omit ``assignment``
entirely for uniform allocation across all variants.


What variants can differ in
---------------------------

**Free to vary across variants:** the mesh asset assigned to each
slot, the number of mesh geoms per ``(body, role)`` bucket on the
variant body (one variant can have more collision meshes than
another), the per-mesh-geom attributes that travel with the mesh
(friction, contact bits, mass, density, ``condim``, and a handful of
others), and explicit body inertial values within whichever single
inertial mode the variants agree on per body.

**Must match across variants:** the body tree, joint topology,
primitive (non-mesh) geoms, and any actuators / sensors / tendons /
equalities. Variants must also agree on the inertial representation
per body (mesh-derived, diagonal, or fullinertia), and may not use the
reserved ``mjlab/pad/`` name prefix on any element. Variant entities
must also be floating-base: the root body declares a freejoint.

The validator runs at entity build time and raises ``ValueError``
naming the offending variant and the exact mismatch.


How variants are assembled
--------------------------

mjlab merges every variant's mesh assets into a single ``MjSpec`` and
gives the variant body enough mesh-geom *slots* to cover the maximum
mesh count any variant uses for each ``(body, role)`` bucket. A slot
is identified by ``(body_path, role, ordinal)``. ``role`` is "visual"
or "collision", derived from ``contype``/``conaffinity``;
mujoco_warp's ``geom_contype``/``geom_conaffinity`` are 1D shared
(not per-world), so a slot's role is fixed across worlds by
construction.

A worked example
~~~~~~~~~~~~~~~~

Say variant ``sphere`` has 1 visual mesh geom and 2 collision mesh
geoms on the prop body, and variant ``cone`` has 1 visual mesh geom
and 4 collision mesh geoms on the same body.

.. code-block:: text

    sphere variant body              cone variant body
    -------------------              -------------------
    prop body                        prop body
      [visual] sphere_vis              [visual] cone_vis
      [coll]   sphere_col_0            [coll]   cone_col_0
      [coll]   sphere_col_1            [coll]   cone_col_1
                                       [coll]   cone_col_2
                                       [coll]   cone_col_3

mjlab walks each variant's body tree, buckets mesh geoms by
``(body_path, role)``, and lays the union out as slots:

.. list-table::
   :header-rows: 1
   :widths: 8 18 8 12 27 27

   * - Slot
     - body_path
     - role
     - ordinal
     - sphere fills with
     - cone fills with
   * - 0
     - /prop
     - visual
     - 0
     - sphere_vis
     - cone_vis
   * - 1
     - /prop
     - collision
     - 0
     - sphere_col_0
     - cone_col_0
   * - 2
     - /prop
     - collision
     - 1
     - sphere_col_1
     - cone_col_1
   * - 3
     - /prop
     - collision
     - 2
     - *(unfilled)*
     - cone_col_2
   * - 4
     - /prop
     - collision
     - 3
     - *(unfilled)*
     - cone_col_3

Five slots total. The merged scene's prop body has five mesh geoms:
slot 0 plus four collision slots (the union of sphere's two and
cone's four). At merge time, every variant's mesh asset is added to
the merged spec under a unique name (e.g.
``sphere/sphere_vis``, ``cone/cone_col_2``).

The merged scene compiles once into a single canonical ``MjModel``
that every world in the batch agrees on layout-wise: same nbody,
ngeom, same body and geom IDs. mjlab's per-world overrides on top of
that one model are what make worlds heterogeneous.

What each world sees at runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Worlds where ``sphere`` is active see only its three meshes; the two
extra collision slots are disabled via per-world ``geom_dataid = -1``,
and mujoco_warp skips them. Worlds where ``cone`` is active see all
five meshes wired up.

.. list-table::
   :header-rows: 1
   :widths: 14 14 12 12 12 12 12

   * - World
     - variant
     - slot 0
     - slot 1
     - slot 2
     - slot 3
     - slot 4
   * - 0
     - sphere
     - sphere_vis
     - sphere_col_0
     - sphere_col_1
     - **off (-1)**
     - **off (-1)**
   * - 1
     - cone
     - cone_vis
     - cone_col_0
     - cone_col_1
     - cone_col_2
     - cone_col_3

Three categories of per-world override carry the variation:

* **geom_dataid** is a ``(num_envs, ngeom)`` table. Its row for
  world W picks which compiled mesh each slot points at. ``-1`` is
  the "skip me" sentinel mujoco_warp already understands.
* **Mesh-derived fields** (``geom_size``, ``geom_rbound``,
  ``geom_aabb``, ``geom_pos``, ``geom_quat``, ``body_mass``,
  ``body_subtreemass``, ``body_inertia``, ``body_invweight0``,
  ``body_ipos``, ``body_iquat``) are stored as ``(num_envs, ...)``
  arrays. The values for sphere worlds reflect a sphere-shaped
  inertia tensor and sphere-sized AABBs; the values for cone worlds
  reflect the cone. The full list is in
  ``mjlab.entity.variants.VARIANT_DEPENDENT_FIELDS``.
* **Per-mesh-geom attributes** (contact bits, friction, mass,
  density, condim, group, priority, rgba, solref, solimp, margin,
  gap) are captured per variant in ``VariantGeomSpec`` at merge time
  and restored verbatim on the slot geom during the per-variant
  reference compile. So if sphere's collision geoms have
  ``friction=0.5`` and cone's have ``friction=1.2``, world W's
  per-step friction reflects the assigned variant's source value.
  The one exception is ``material``, which is not propagated across
  variants; if you need per-world appearance variation use DR on
  ``geom_rgba`` / ``mat_rgba``.

If ``sphere`` adds a body that ``cone`` lacks (or vice versa), the
validator rejects the configuration before any of the merge logic
runs. The slot mechanism only flexes mesh geom counts within
matching bodies; everything structural above the geom level must
agree.

.. note::

   **Doesn't compiling the merged scene ruin the prop body's
   inertia?**

   No, but it's worth understanding why, because the naive intuition
   says it should. If you stuck every variant's mesh geoms on the
   prop body and called ``spec.compile()``, MuJoCo would sum each
   geom's inertial contribution, and you would get a body whose mass
   and inertia tensor are a meaningless mix of every variant's shape.

   mjlab avoids this in two layers:

   * **The merged scene does not stick every variant's geoms on the
     body.** The prop body in the merged spec carries variant 0's
     mesh geoms (with their original mass and density) plus, for any
     slot variant 0 doesn't fill, a synthesized padding geom that has
     ``mass = 0`` and ``density = 0``. Padding contributes nothing to
     body inertia. Other variants' meshes are present in the merged
     spec only as **mesh assets** (in the assets section, not as geoms
     on any body). They get wired in at runtime via per-world
     ``geom_dataid`` and never affect the host compile's inertial
     sums.
   * **Per-world overrides come from per-variant source compiles.**
     Even with the above, the merged-scene compile's prop body inertia
     is only correct for variant 0. For every other variant, mjlab
     compiles that variant's original source spec in isolation (one
     body, one variant's worth of meshes), reads the resulting
     ``body_mass``, ``body_inertia``, ``body_ipos``, ``body_iquat``,
     ``body_invweight0``, and ``body_subtreemass``, and writes them
     into the per-world arrays at the prop body's index.

   Net result: world W's prop body inertia is byte-equal to what you
   would get by compiling variant W's source spec on its own. There
   is a regression test
   (``test_visual_collision_split_inertia_matches_independent_compile``
   in ``tests/test_variants.py``) that asserts exactly this against
   independent per-variant compiles.


World assignment
----------------

How worlds get mapped to variants is controlled by the ``assignment``
field on ``VariantEntityCfg``. It accepts three shapes:

* ``None`` (default): uniform allocation across variants.
* ``dict[str, float]``: per-variant weights. Variants not listed
  default to weight 1.0.
* ``Callable[[int], Sequence[int]]``: an explicit assignment function
  called with ``num_envs`` at simulation init.

Both the ``None`` and dict cases use the
`largest remainder method
<https://en.wikipedia.org/wiki/Largest_remainder_method>`_. Each
variant's quota is ``q_i = (w_i / sum(w)) * num_envs``; each variant
first receives ``floor(q_i)`` worlds, and the remaining
``num_envs - sum(floors)`` worlds go to the variants with the largest
fractional remainders, with ties broken by declaration order. For
``num_envs = 10`` and weights ``(1.0, 2.0, 1.0)`` this gives
``(3, 5, 2)`` worlds per variant. Weights are normalized internally,
so ``{"a": 1, "b": 2, "c": 1}`` and ``{"a": 0.25, "b": 0.5, "c": 0.25}``
produce identical assignments. A weight of zero is allowed and
produces zero worlds for that variant; at least one variant must end
up with positive weight.

The default and dict paths are purely deterministic given
``(assignment, num_envs)``. With ``assignment={"a": 1, "b": 1}`` and
``num_envs = 8`` you always get ``[0, 0, 0, 0, 1, 1, 1, 1]``. There is
no seed involved; rerunning the same config produces the same
partition every time. Note that the partition's *boundaries* depend
on ``num_envs``, so world W's variant is not necessarily stable when
you change ``num_envs``. If you need explicit per-world stability
across batch sizes (e.g. "world 0 is always variant 0, world 1 is
always variant 1, regardless of how many envs I launch"), use a
callable assignment as below.

Variant assignment is fixed at ``Simulation`` initialization and does
not resample on episode reset. The intended use is heterogeneous
training across the batch, not per-episode mesh randomization.

Read the resolved assignment from user code via
``env.sim.world_to_variant``:

.. code-block:: python

    >>> env.sim.world_to_variant["object"]
    tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

The mapping is keyed by entity name (without trailing slash) and
returns a ``(num_envs,)`` tensor of variant indices in the order
variants were declared in ``VariantEntityCfg.variants``. The dict is
empty for non-variant scenes.


Custom assignment with a callable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the weighted default is not what you want, pass a callable to
``assignment``. The callable receives ``num_envs`` and must return a
length-``num_envs`` sequence of variant indices in
``[0, len(variants))``. The returned sequence's length and bounds are
validated at sim init; mismatches raise a ``ValueError`` naming the
offending entity.

A few patterns:

**Round-robin** - cycle through variants by world index.

.. code-block:: python

    cfg = VariantEntityCfg(
        variants={"a": make_a, "b": make_b, "c": make_c},
        assignment=lambda n: [w % 3 for w in range(n)],
    )

**Stratified halves** - first half is variant 0, second half is
variant 1.

.. code-block:: python

    cfg = VariantEntityCfg(
        variants={"easy": make_easy, "hard": make_hard},
        assignment=lambda n: [0] * (n // 2) + [1] * (n - n // 2),
    )

Domain randomization
--------------------

Domain randomization on variant scenes preserves per-variant baselines
automatically. When the simulation initializes, mjlab snapshots the
variant-dependent fields as ``(num_envs, ...)`` tensors and registers
them in ``sim.per_world_default_fields``. DR operations that read
defaults (scale, additive offsets) detect this registration and index
the per-world default array by environment, so a 10% mass scale
applied across a batch containing a 100 g sphere variant and a 1 kg
cube variant produces 10% perturbations *around each variant's own
mass*, not 10% of a shared template mass.

Fields that are not variant-dependent (``geom_friction``,
``dof_armature``, ``dof_damping``, and so on) behave identically on
variant and non-variant scenes.

For inertial randomization the recommended path is
``dr.pseudo_inertia``, which jointly randomizes mass, COM offset,
principal moments of inertia, and principal frame orientation through
the pseudo-inertia matrix factorization of `Rucker and Wensing (2022)
<https://par.nsf.gov/servlets/purl/10347458>`_. It is exact for any
perturbation magnitude and remains physically consistent across
variants of different scale. ``dr.body_mass`` modifies ``body_mass``
without touching the inertia tensor and emits a ``UserWarning`` when
called; it is appropriate only for modeling a point mass added at the
COM, not for density-like randomization. The distinction matters more
on variant scenes than on single-asset scenes because variants often
differ in mass by an order of magnitude.


Viewers
-------

The native viewer, offscreen renderer, and Viser viewer all sync the
selected environment's per-world fields into the host ``MjModel``
before rendering, so the rendered geometry matches the variant
assigned to the viewed environment. Switching environments in the
native viewer (the ``,`` and ``.`` keys) updates the displayed mesh
accordingly.

Viser bakes mesh data into batched handles and cannot rely on a live
view of ``geom_dataid``. It groups worlds by visual fingerprint (mesh
selection, local geom frames, baked appearance) and builds one batched
handle per group, with each environment assigned to its handle. A
scene with N variants typically produces up to N handles per body.
Convex hull visualization is computed per variant from the variant's
mesh vertices.


Performance
-----------

**Per-step cost is unaffected by variant count.** Variant-dependent
fields are stored as per-world arrays accessed by world index in the
existing kernels, with no branching or dispatch on variant.

**Construction cost is linear in the total variant count.** mjlab
compiles the merged scene once to produce the canonical ``MjModel``,
then compiles each variant's original (un-merged) source spec in
isolation to recover that variant's per-body and per-geom mesh-derived
fields. Each per-variant compile sees only that variant's single body
and mesh, so its cost is independent of the total number of variants
in the scene.

For a scene with one variant entity declaring k variants, construction
runs ``1 + k`` compiles. With multiple variant entities, compiles
decouple across entities: two variant entities of 5 variants each cost
``1 + 5 + 5 = 11`` compiles, not ``1 + 5 * 5 = 26``. As an order of
magnitude on CPU with typical procedural meshes, each per-variant
compile takes around 1-2 ms, so a scene with 100 variants pays a few
hundred milliseconds at startup and a scene with 1000 variants pays
roughly two seconds.

The merged spec contains every variant's mesh assets simultaneously,
so memory at scene-build time scales with the total mesh vertex /
face count across all variants. This is paid once at startup and does
not affect training throughput.


Limitations
-----------

**Floating-base only.** Each variant's root body must declare a free
joint. Fixed-base variants are rejected; mocap auto-wrapping that
applies to non-variant entities is not applied here.

**Material assets are not propagated.** Each variant's ``contype``,
``conaffinity``, ``condim``, ``friction``, ``mass``, ``density``,
``group``, ``priority``, ``rgba``, ``solref``, ``solimp``, ``margin``,
and ``gap`` are restored per-world during compile, but the
``material`` reference on slot geoms inherits whichever material the
template variant set. Use DR on ``geom_rgba`` / ``mat_rgba`` for
per-world appearance variation.

**Assignment is fixed at sim init.** There is no API to swap a world
to a different variant on episode reset. World W's mesh asset is
whatever it was assigned at init for the lifetime of the simulation.
Per-episode mesh randomization is not supported today; DR can vary
scalar properties (mass, friction, color, scale) on a fixed variant
but cannot swap one mesh for another.

**No support for per-world differing kinematic topology.** Variants
must share the same body tree, joints, and actuator/sensor counts,
so you cannot configure things like:

* a different number of objects per world (world 0 has two props on
  the table, world 1 has three);
* different articulation per world (world 0's prop is an articulated
  drawer with a slider joint, world 1's prop is a rigid block).

True heterogeneous topology requires upstream support in mujoco_warp
that does not currently exist.
