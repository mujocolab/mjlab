.. _per_world_mesh:

Mesh Variants
=============

Mesh variants let a single batched simulation run with different mesh
assets in different parallel worlds. World 0 may simulate a cube, world
1 a sphere, and world 2 a bowl, all sharing the same compiled scene
and the same kinematic structure. The result is a heterogeneous batch
in which the mesh and its derived constants vary across worlds while
everything else (the body tree, the joint structure, the contact and
solver setup) is fixed.

Mesh variants are configured at the entity level through
``VariantEntityCfg`` and ``VariantCfg``. Once configured,
domain randomization, the native viewer, the offscreen renderer, and
the Viser viewer all pick up the variant assignment automatically.


How it works
------------

A standard ``EntityCfg`` provides a single ``spec_fn`` that returns one
``MjSpec``. A ``VariantEntityCfg`` provides a dictionary of named
variants, each with its own ``spec_fn`` and a weight controlling the
proportion of worlds that use it.

**All variants must declare the same kinematic structure.** The batched
simulator assumes a single topology across worlds; per-world variation
is confined to mesh assets and the constants derived from them. mjlab
uses the first variant's body tree as the template and copies mesh
assets and explicit body inertials from the others. Geom-level
properties on later variants such as ``rgba``, friction, and material
assignments are not propagated; control per-world appearance through
domain randomization on ``geom_rgba`` or ``mat_rgba``. The structural
check is enforced at construction time and raises a ``ValueError``
describing the first mismatch. Variants must also be floating-base
(declare a free joint on the root body); fixed-base variants are
rejected.

A minimal two-variant config:

.. code-block:: python

    import mujoco

    from mjlab.entity import EntityCfg, VariantCfg, VariantEntityCfg

    def make_sphere_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        mesh = spec.add_mesh(name="visual")
        mesh.make_sphere(subdivision=3)
        mesh.scale[:] = (0.05,) * 3
        body = spec.worldbody.add_body(name="prop")
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="visual")
        return spec

    # ``make_cone_spec`` follows the same shape with
    # ``mesh.make_cone(nedge=16, radius=0.04)`` in place of the sphere call.

    object_cfg = VariantEntityCfg(
        variants={
            "sphere": VariantCfg(spec_fn=make_sphere_spec, weight=1.0),
            "cone": VariantCfg(spec_fn=make_cone_spec, weight=2.0),
        },
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    )

During scene construction mjlab merges the per-variant specs into a
single ``MjSpec`` whose mesh slots are padded to the maximum count any
variant uses, then writes a per-world ``geom_dataid`` table that
selects the right mesh for each world. In the merged scene
``geom_dataid`` is no longer a flat ``(ngeom,)`` vector but a
``(num_envs, ngeom)`` table whose rows differ by variant. A value of
``-1`` marks a disabled mesh slot, used for variants with fewer mesh
geoms than the maximum.

Mesh choice is entangled with several other compiled-model constants:
geom collision bounds, geom local frames, body inertials, subtree mass,
and inverse weights. mjlab compiles each unique row of the
``geom_dataid`` table on the host and copies the relevant compiled
fields into per-world arrays on the GPU, so each world's compiled
constants stay consistent with that world's mesh selection. The full
list of fields handled this way is in
``mjlab.sim.mesh_variants.VARIANT_DEPENDENT_FIELDS``.


World assignment
----------------

mjlab assigns variants to worlds proportionally by weight using the
`largest remainder method
<https://en.wikipedia.org/wiki/Largest_remainder_method>`_. Each
variant's quota is ``q_i = (w_i / sum(w)) * num_envs``; each variant
first receives ``floor(q_i)`` worlds, and the remaining
``num_envs - sum(floors)`` worlds go to the variants with the largest
fractional remainders, with ties broken by declaration order. For
``num_envs = 10`` and weights ``(1.0, 2.0, 1.0)`` this gives
``(3, 5, 2)`` worlds per variant. Weights are normalized internally,
so ``(1, 2, 1)`` and ``(0.25, 0.5, 0.25)`` produce identical
assignments. A weight of zero is allowed and produces zero worlds for
that variant; at least one variant must have a positive weight.

Variant assignment is fixed at simulation initialization and does not
resample on episode reset. The intended use is heterogeneous training
across the batch, not per-episode mesh randomization. To inspect the
assignment from user code, read ``env.sim.world_to_variant``:

.. code-block:: python

    >>> env.sim.world_to_variant["object"]
    tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

The mapping is keyed by entity name (without trailing slash) and
returns a ``(num_envs,)`` tensor of variant indices in the order
variants were declared in ``VariantEntityCfg.variants``. The dict is
empty for non-variant scenes.


Domain randomization
--------------------

Domain randomization on variant scenes preserves per-variant baselines
automatically. When the simulation initializes, mjlab snapshots the
variant-dependent fields (``body_mass``, ``body_inertia``,
``geom_size``, and others listed in ``VARIANT_DEPENDENT_FIELDS``) as
``(num_envs, ...)`` tensors and registers them in
``sim.per_world_default_fields``. Domain randomization operations that
read defaults (scale, additive offsets) detect this registration and
index the per-world default array by environment, so a 10% mass scale
applied across a batch containing a 100 g sphere variant and a 1 kg
cube variant produces 10% perturbations around each variant's own
mass, not 10% of a shared template mass. Fields that are not
variant-dependent (``geom_friction``, ``dof_armature``,
``dof_damping``, and so on) behave identically on variant and
non-variant scenes.

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


Performance considerations
--------------------------

Mesh variants do not add per-step overhead in the GPU kernels.
Variant-dependent fields are stored as per-world arrays accessed by
world index in the existing kernels, with no branching or dispatch
on variant.

Initialization is the main consideration. mjlab compiles each unique
row of the ``geom_dataid`` table by taking a fresh ``MjSpec.copy()``,
editing the mesh selection and (if applicable) the explicit body
inertials, and calling ``spec.compile()``. This work scales with the
number of unique variant combinations rather than with ``num_envs``.
For a scene with one variant entity declaring k variants, this is k
host compiles regardless of how many worlds use each variant. With
multiple variant entities the unique-row count is bounded by the
product of their variant counts in the worst case, so a scene with
two variant entities of 5 variants each could trigger up to 25 host
compiles at init.

``MjSpec.copy()`` and ``spec.compile()`` are non-trivial operations,
and their cost grows with scene size. For a scene with many variant
entities or many variants per entity, the cumulative initialization
cost can be measured in seconds. This cost is paid once at startup
and does not affect training throughput.

The merged spec contains every variant's mesh assets simultaneously.
Memory footprint at scene-build time scales with the total number of
mesh vertices and faces across all declared variants.
