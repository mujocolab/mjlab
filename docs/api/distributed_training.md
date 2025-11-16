# Distributed Training

mjlab supports multi-GPU distributed training using `torchrunx`. Distributed
training parallelizes RL workloads across multiple GPUs by running independent
rollouts on each device and synchronizing gradients during policy updates.
Throughput scales nearly linearly with GPU count.

## TL;DR

Launch distributed training using `--num-gpus`:

```bash
uv run train <task-name> \
  --num-gpus N \
  <task-specific CLI args>
```

**Key points:**
- `--num-gpus N` automatically spawns N processes (one per GPU). So for 4 GPUs,
  use `--num-gpus 4`, for 2 GPUs, use `--num-gpus 2`, etc.
- Devices are automatically assigned (`cuda:0`, `cuda:1`, ..., `cuda:N-1`)
- Do not specify `--device` with `--num-gpus` (single-GPU only)
- Each GPU runs the full `num-envs` count (e.g., 2 GPUs × 4096 envs = 8192
  total)

## How It Works

mjlab's role is simple: **isolate mjwarp simulations on each GPU** using
`wp.ScopedDevice`. This ensures each process's environments stay on their
assigned device. `torchrunx` handles the rest.

**Process spawning.** `torchrunx` spawns N independent processes (one per GPU)
and sets environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) to
coordinate them. Each process executes the full training script with its
assigned GPU.

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
