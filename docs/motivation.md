# Why mjlab?

## The Problem with Current Tools

The GPU-accelerated robotics simulation landscape has some great tools, but each has significant pain points:

### IsaacLab: Great API, Terrible UX
- **Heavy Installation**
- **Slow Startup**
- **Black Box Simulator**
- **Omniverse Overhead**

### MJX: JaX Painpoints
- **Poor collision scaling**
- **JaX higher learning curve**

## What about Newton?

## Our Solution: mjlab

**mjlab = IsaacLab's proven API + MuJoCo's simplicity + GPU acceleration**

We took the best parts of IsaacLab (manager-based architecture, RL abstractions) and built them on top of MuJoCo Warp for maximum performance and transparency.

## Why Not Just Add MuJoCo Warp to IsaacLab?

This was actually our first instinct! But after investigation:

1. **Architectural Mismatch**: IsaacLab is deeply integrated with Omniverse/Isaac Sim
2. **Performance**: Omniverse overhead remains even with MuJoCo backend
3. **Complexity**: Supporting multiple simulators adds maintenance burden
4. **Focus**: IsaacLab serves many domains; we can optimize for humanoids

Starting fresh let us build exactly what the subset of the community wants.

## Our Philosophy

### Bare Metal Performance
- Direct MuJoCo Warp access, no translation layers
- Exposed mjModel/mjData structs for maximum control
- GPU-accelerated simulation with minimal overhead

### Developer Experience First
- One-line installation with `uv`
- Sub-5-second startup times
- Kernel caching

### Focused Scope
- Target: Two proven robots (Unitree G1, Go1) + RL framework
- Not trying to be everything to everyone

## The Bottom Line

If you're working on legged robot RL and want:
- Fast iteration cycles
- Full physics control
- Proven RL abstractions
- GPU acceleration
- Simple installation
- Love MuJoCo and its community

Then mjlab is built for you.

If you need manipulation, complex sensors, or tight USD pipeline integration, stick with IsaacLab.

The goal isn't to replace IsaacLab everywhere – it's to provide the best possible experience for legged robot researchers and practitioners.
