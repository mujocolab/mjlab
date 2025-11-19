# Distributed Training

mjlab supports multi-GPU distributed training using
[torchrunx](https://github.com/apoorvkh/torchrunx). Distributed training
parallelizes RL workloads across multiple GPUs by running independent rollouts
on each device and synchronizing gradients during policy updates. Throughput
scales nearly linearly with GPU count.

## TL;DR

**Single GPU (default):**
```bash
uv run train <task-name> <task-specific CLI args>
# or explicitly: --gpu-ids 0
```

**Multi-GPU:**
```bash
uv run train <task-name> \
  --gpu-ids 0 1 \
  <task-specific CLI args>
```

**All GPUs:**
```bash
uv run train <task-name> \
  --gpu-ids all \
  <task-specific CLI args>
```

**CPU mode:**
```bash
uv run train <task-name> \
  --gpu-ids None \
  <task-specific CLI args>
# or: CUDA_VISIBLE_DEVICES="" uv run train <task-name> ...
```

**Key points:**
- `--gpu-ids` specifies GPU indices (e.g., `--gpu-ids 0 1` for 2 GPUs)
- GPU indices are relative to `CUDA_VISIBLE_DEVICES` if set
- `CUDA_VISIBLE_DEVICES=2,3 uv run train ... --gpu-ids 0 1` uses physical GPUs 2 and 3
- Each GPU runs the full `num-envs` count (e.g., 2 GPUs × 4096 envs = 8192 total)
- Single-GPU and CPU modes run directly; multi-GPU uses `torchrunx` for process
  spawning

## Configuration

**torchrunx Logging:**

By default, torchrunx process logs are saved to `{log_dir}/torchrunx/`. You can
customize this:

```bash
# Disable torchrunx file logging.
uv run train <task-name> --gpu-ids 0 1 --torchrunx-log-dir ""

# Custom log directory.
uv run train <task-name> --gpu-ids 0 1 --torchrunx-log-dir /path/to/logs

# Or use environment variable (takes precedence over flag).
TORCHRUNX_LOG_DIR=/tmp/logs uv run train <task-name> --gpu-ids 0 1
```

The priority is `TORCHRUNX_LOG_DIR` env var, `--torchrunx-log-dir` flag, default
`{log_dir}/torchrunx`.

**Single-Writer Operations:**

Only rank 0 performs file I/O operations (config files, videos, wandb logging)
to avoid race conditions. All workers participate in training, but logging
artifacts are written once by the main process.

## How It Works

mjlab's role is simple: **isolate mjwarp simulations on each GPU** using
`wp.ScopedDevice`. This ensures each process's environments stay on their
assigned device. `torchrunx` handles the rest.

**Process spawning.** Multi-GPU training uses `torchrunx.Launcher(...).run(...)`
to spawn N independent processes (one per GPU) and sets environment variables
(`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) to coordinate them. Each process executes
the training function with its assigned GPU.

**Independent rollouts.** Each process maintains its own:
- Environment instances (with `num-envs` parallel environments), isolated on
  its assigned GPU via `wp.ScopedDevice`
- Policy network copy
- Experience buffer (sized `num_steps_per_env × num-envs`)

Each process uses `seed = cfg.seed + local_rank` to ensure different random
experiences across GPUs, increasing sample diversity.

**Gradient synchronization.** During the update phase, `rsl_rl` synchronizes
gradients after each mini-batch through its `reduce_parameters()` method:
1. Each process computes gradients independently on its local mini-batch
2. All policy gradients are flattened into a single tensor
3. `torch.distributed.all_reduce` averages gradients across all GPUs
4. Averaged gradients are copied back to each parameter, keeping policies
   synchronized
