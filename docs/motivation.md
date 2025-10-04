# Why mjlab?

## The Problem

GPU-accelerated robotics simulation has great tools, but each has tradeoffs:

**Isaac Lab**: Excellent API and RL abstractions, but heavy installation, slow startup, and Omniverse overhead make rapid iteration painful.

**MJX**: Fast and lightweight, but JAX's learning curve and poor collision scaling (if using the `'jax'` [implementation](https://github.com/google-deepmind/mujoco/blob/32e08f9507c9bdc5a1a5411c6fa9f0346542b038/mjx/mujoco/mjx/_src/types.py#L28-L33) rather than the `'warp'` one) limit adoption.

**Newton**: Brand new generic simulator supporting multiple solvers (MuJoCo, VBD, etc.) with USD-based format instead of MJCF/XML. Doesn't yet have the ecosystem and community resources that MuJoCo has built over the years.

## Our Solution

**mjlab = Isaac Lab's API + MuJoCo's simplicity + GPU acceleration**

We took Isaac Lab's proven manager-based architecture and RL abstractions, then built them directly on MuJoCo Warp. No translation layers, no Omniverse overhead. Just fast, transparent physics.

### Why Not Use Isaac Lab with Newton?

Isaac Lab recently added [experimental Newton support](https://github.com/isaac-sim/IsaacLab/tree/dev/newton), which is great for existing Isaac users who want to try MuJoCo via Newton's backend.

If you want a comprehensive platform (RL, imitation learning, photorealistic rendering, etc.), use Isaac Lab. If you want a focused tool for RL and sim2real with MuJoCo, use mjlab.

### Why Not Add MuJoCo Warp to Isaac Lab?

This would be fantastic for the ecosystem! NVIDIA's team is exploring this with their recent [experimental Newton integration](https://github.com/isaac-sim/IsaacLab/tree/dev/newton), which is exciting.

But for us, we wanted to start with something more focused that we could realistically maintain. Isaac Lab is architected around Omniverse/Isaac Sim's powerful capabilities, which makes sense given everything it supports. Integrating MuJoCo Warp there would mean working within that broader framework and supporting use cases beyond our scope.

Maintaining multi-backend compatibility naturally involves tradeoffs in complexity and dependency management. By starting fresh, we could:
- Write a lean codebase optimized specifically for MuJoCo Warp
- Keep dependencies minimal and installation fast
- Maintain direct access to native mjModel/mjData structures
- Iterate quickly without navigating a larger platform's constraints

Think of mjlab as a love letter to Isaac Lab's brilliant API design. We're bringing those manager-based abstractions to researchers who want something smaller and MuJoCo-specific. It's complementary, not competitive.

## Philosophy

**Bare Metal Performance**
- Direct MuJoCo Warp integration, no translation layers
- Native mjModel/mjData structures MuJoCo users know and love
- GPU-accelerated with minimal overhead

**Developer Experience First**
- One-line installation: `uvx --from mjlab demo`
- Blazing fast startup
- Standard Python debugging (pdb anywhere!)
- Fast iteration cycles

**Focused Scope**
- Rigid-body robotics and RL, not trying to do everything
- Clean, maintainable codebase over feature bloat
- MuJoCo-native implementation, not a generic wrapper

## When to Use mjlab

**Use mjlab if you want:**
- Fast iteration and debugging
- Direct MuJoCo physics control
- Proven RL abstractions (Isaac Lab-style)
- GPU acceleration without heavyweight dependencies
- Simple installation and deployment

**Use Isaac Lab if you need:**
- Photorealistic rendering
- USD pipeline integration
- Omniverse ecosystem features

**Use Newton if you need:**
- Multi-physics solver support (e.g., deformables)
- Differentiable simulation

## The Bottom Line

mjlab isn't trying to replace everything. It's built for researchers who love MuJoCo's simplicity and want Isaac Lab's RL abstractions with GPU acceleration, minus the overhead.
