# mjlab

This is extremely bad code and very much work in progress. I'm just exploring different design patterns for physics simulation + RL workflows.

## Installation

```bash
uv pip install -e .
python -c "import mjlab"  # Trigger menagerie download.
```

## Benchmark

```bash
python mjlab/rl/train.py --num-envs 128 --agent-cfg.experiment-name cartpole_experiment --agent-cfg.max-iterations 1500
```

For cpu, add ` --agent-cfg.device "cpu"`.
