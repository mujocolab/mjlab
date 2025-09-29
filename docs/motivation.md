# Why mjlab?

## The Problem

GPU-accelerated robotics simulation has great tools, but each has tradeoffs:

**Isaac Lab**: Excellent API and RL abstractions, but heavy installation, slow startup, and Omniverse overhead make rapid iteration painful.

**MJX**: Fast and lightweight, but JAX's learning curve and poor collision scaling limit adoption.

**Newton**: Promising generic simulator supporting multiple solvers (MuJoCo, VBD, etc.), but the generic API requires everything to be converted into Newton's format, adding a translation layer. Also brand new, so it doesn't yet have the vibrant ecosystem and community resources that MuJoCo and ISAAC have built over the years.

## Our Solution

**mjlab = Isaac Lab's API + MuJoCo's simplicity + GPU acceleration**

We took Isaac Lab's proven manager-based architecture and RL abstractions, then built them directly on MuJoCo Warp. No translation layers, no Omniverse overhead—just fast, transparent physics.

### Why Not Just Add MuJoCo Warp to Isaac Lab?

We explored this first! But:
- Isaac Lab is deeply integrated with Omniverse/Isaac Sim's architecture
- Omniverse has high overhead
- Heavy dependency stack from trying to support many use cases
- Supporting multiple backends adds complexity and maintenance burden
- Starting fresh let us write lean, performant code

Isaac Lab recently added experimental Newton support, but maintaining compatibility across multiple simulators and backends inherently limits how deeply you can optimize for any single one. We chose focus over flexibility.

## Philosophy

**Bare Metal Performance**
- Direct MuJoCo Warp integration—no translation layers
- Native mjModel/mjData structures MuJoCo users know and love
- GPU-accelerated with minimal overhead

**Developer Experience First**
- One-line installation: `uvx --from mjlab demo`
- Instant startup (no compilation, no Omniverse loading)
- Standard Python debugging (pdb anywhere!)
- Fast iteration cycles

**Focused Scope**
- Rigid-body robotics and RL—not trying to do everything
- Clean, maintainable codebase over feature bloat
- Direct MuJoCo integration over generic abstractions

## When to Use `mjlab`

**Use `mjlab` if you want:**
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
- Deformable object simulation
- Multi-physics solver support
- Generic simulator abstraction

## The Bottom Line

mjlab isn't trying to replace everything—it's built for researchers who love MuJoCo's simplicity and want Isaac Lab's RL abstractions with GPU acceleration, minus the overhead.

If that's you, welcome home.
