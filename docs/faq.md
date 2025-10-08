# FAQ & Troubleshooting

## Platform Support

### Does it work on macOS?

Yes, but only with limited performance. mjlab runs on macOS using CPU-only execution through MuJoCo Warp.
- **Training is not recommended on macOS** as it lacks GPU acceleration.
- **Evaluation works**, but is significantly slower than on Linux with CUDA.
We recommend Linux with an NVIDIA GPU for serious training workloads.

### Does it work on Windows?

We have not tested on Windows. Community contributions for Windows support are welcome!

### CUDA Compatibility

Not all CUDA versions are supported by MuJoCo Warp.
- See [mujoco_warp#101](https://github.com/google-deepmind/mujoco_warp/issues/101) for details.
- **Recommended**: CUDA 12.4+ (for conditional execution in CUDA graphs).

---

## Performance

### Is it faster than Isaac Lab?

On par or faster based on our experience over the last few months.

### What GPU do you recommend?

- RTX 40-series GPUs (or newer)
- L40s, H100

---

## Rendering & Visualization

### What visualization options are available?

We currently support two visualizers for policy evaluation and debugging:
- **Native MuJoCo visualizer** - The built-in visualizer that ships with MuJoCo
- **[Viser](https://github.com/nerfstudio-project/viser)** - Web-based 3D visualization

We’re exploring options for **training-time visualization** (e.g. live rollout
viewers), but this is not yet available. As a current alternative, mjlab
supports **video logging to Weights & Biases (W&B)**, so you can monitor rollout
videos directly in your experiment dashboard.

### What about camera/pixel rendering for vision-based RL?

Camera rendering for pixel-based agents is not yet available. The MuJoCo Warp
team is actively developing camera support, which will integrate with mjlab once
available.

---

## Assets & Compatibility

### What robots are included?

mjlab includes two reference robots:
- **Unitree Go1** (quadruped)
- **Unitree G1** (humanoid)

These serve as examples for robot integration and support our reference tasks for testing. We intentionally keep mjlab lean and lightweight, so we don't plan to expand the built-in robot library. Additional robots may be provided in a separate repository.

### Can I use USD or URDF models?

No, mjlab requires MJCF (MuJoCo XML) format. You'll need to convert USD/URDF models to MJCF. Fortunately, [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) provides a large collection of pre-converted robot assets you can use.

---

## Getting Help

### GitHub Issues

**For bug reports**, please include:
- CUDA driver version
- GPU model
- Minimal reproduction script
- Complete error logs and stack traces
- Appropriate tags: `bug`, `performance`, `docs`

[Open an issue →](https://github.com/mujocolab/mjlab/issues)

### Discussions

**For usage questions** (config, performance tips, asset conversion, best practices):
[Start a discussion →](https://github.com/mujocolab/mjlab/discussions)

### Contributing

**Want to help improve mjlab?**
- Bug fixes and performance optimizations
- Feature implementations (check issues tagged `enhancement`)
- Documentation improvements

Before contributing:
- Run `make test` to ensure changes don’t break existing functionality
- Run `make format` or use `pre-commit` hooks for code style consistency

---

## Known Limitations

We're tracking missing features for the stable release in
[issue #100](https://github.com/mujocolab/mjlab/issues/100). Check our
[open issues](https://github.com/mujocolab/mjlab/issues) to see what's actively
being worked on.

If something isn't working or if we've missed something, please
[file a bug report](https://github.com/mujocolab/mjlab/issues/new).

> **Reminder**: mjlab is in **beta**. Breaking changes and missing features are
> expected — feedback and contributions are welcome!
