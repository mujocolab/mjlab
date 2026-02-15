# mjlab Nightly Benchmarks

This directory contains scripts for automated nightly benchmarking of mjlab.

## Overview

The nightly benchmark system:
1. Trains a tracking policy on the latest commit
2. Evaluates the policy across 1024 trials
3. Measures simulation throughput
4. Generates an HTML report with historical trends
5. Publishes results to GitHub Pages

## Usage

### Run the full nightly benchmark

```bash
./scripts/benchmarks/nightly_train.sh
```

### Skip training (regenerate report only)

```bash
SKIP_TRAINING=1 ./scripts/benchmarks/nightly_train.sh
```

### Skip training and throughput

```bash
SKIP_TRAINING=1 SKIP_THROUGHPUT=1 ./scripts/benchmarks/nightly_train.sh
```

### Regenerate report directly (no git operations)

```bash
uv run python scripts/benchmarks/generate_report.py \
  --entity gcbc_researchers \
  --tag nightly \
  --output-dir benchmark_results
```

### Measure throughput only

```bash
uv run python scripts/benchmarks/measure_throughput.py \
  --num-envs 4096 \
  --output-dir benchmark_results
```

## Configuration

Environment variables for `nightly_train.sh`:

- `CUDA_DEVICE` - GPU device to use (default: 0)
- `WANDB_TAGS` - Comma-separated tags for the run (default: nightly)
- `SKIP_TRAINING` - Set to "1" to skip training
- `SKIP_THROUGHPUT` - Set to "1" to skip throughput benchmarking

## Automated Setup

See [systemd/README.md](systemd/README.md) for instructions on setting up automated nightly runs using systemd timers.

## Report Options

The `generate_report.py` script supports:

- `--eval-limit N` - Maximum number of NEW runs to evaluate per invocation (default: 10)
  - Set to 0 for no limit
  - Historical cached results are always preserved
- `--tag TAG` - Filter runs by WandB tag (default: "nightly")
- `--num-envs N` - Number of parallel environments for evaluation (default: 1024)

## Viewing Results

Reports are published to: https://mujocolab.github.io/mjlab/nightly/
