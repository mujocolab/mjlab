# FAQ & Troubleshooting

## Platform Support

### Does it work on macOS?

Yes, but with limited performance. mjlab runs on macOS using CPU-only execution through MuJoCo Warp. The macOS wheels for Warp are relatively new, and CPU performance is significantly slower than GPU execution. For serious training, we recommend Linux with CUDA support.

### Does it work on Windows?

We have not tested on Windows. Community contributions for Windows support are welcome!

### CUDA Compatibility

Not all CUDA versions are supported by MuJoCo Warp. Check [mujoco_warp#101](https://github.com/google-deepmind/mujoco_warp/issues/101) for compatibility with your CUDA version.

## Performance

### Is it faster than Isaac Lab?

On par or faster based on our experience over the last few months. We're planning to release a side-by-side performance comparison for both velocity tracking and motion imitation tasks in the near future.

### What GPU do you recommend?

Linux with CUDA support is strongly recommended for training. RTX 40-series GPUs (or newer) provide significantly faster training speeds due to their improved RT cores, which MuJoCo Warp leverages for physics simulation.

## Rendering & Visualization

### What visualization options are available?

We currently support two visualizers for monitoring training:
- **Native MuJoCo visualizer** - The built-in visualizer that ships with MuJoCo
- **[Viser](https://github.com/nerfstudio-project/viser)** - Web-based 3D visualization

### What about camera/pixel rendering for vision-based RL?

Camera rendering for pixel-based agents is not yet available. The MuJoCo Warp team is actively developing camera support, which will integrate with mjlab once available.

## Assets & Compatibility

### What robots are included?

mjlab includes two reference robots:
- **Unitree Go1** (quadruped)
- **Unitree G1** (humanoid)

These serve as examples for robot integration and support our reference tasks for testing. We intentionally keep mjlab lean and lightweight, so we don't plan to expand the built-in robot library. Additional robots may be provided in a separate repository.

### Can I use USD or URDF models?

No, mjlab requires MJCF (MuJoCo XML) format. You'll need to convert USD/URDF models to MJCF. Fortunately, [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) provides a large collection of pre-converted robot assets you can use.

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

**For usage questions:**
- Task configuration help
- Performance optimization tips
- Asset conversion guidance
- Sharing examples and best practices

[Start a discussion →](https://github.com/mujocolab/mjlab/discussions)

### Contributing

**Want to help improve mjlab?**
- Documentation improvements
- Bug fixes and performance optimizations
- Feature implementations (check issues tagged `enhancement`)
- Testing on different platforms

[See CONTRIBUTING.md for guidelines](CONTRIBUTING.md)

## Known Limitations

We're tracking missing features for the stable release in [issue #100](https://github.com/mujocolab/mjlab/issues/100). Check our [open issues](https://github.com/mujocolab/mjlab/issues) to see what's actively being worked on.

If something isn't working or if we've missed something, please [file a bug report](https://github.com/mujocolab/mjlab/issues/new).

**Remember:** mjlab is in beta! We're actively addressing limitations and welcome feedback on priorities.
