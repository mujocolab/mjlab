# mjlab Installation Guide

## System Requirements

- **Python**: 3.10 or higher
- **Operating System**: 
  - Linux (recommended)
  - macOS (limited support - see note below)
  - Windows (untested)
- **GPU**: NVIDIA GPU strongly recommended
  - **CUDA Compatibility**: Not all CUDA versions are supported by MuJoCo Warp
    - Check [mujoco_warp#101](https://github.com/google-deepmind/mujoco_warp/issues/101) for CUDA version compatibility
    - **Recommended**: CUDA 12.4+ (for conditional CUDA graph support)

> **⚠️ Important Note on macOS**: mjlab is designed for training RL policies, which requires a GPU. While macOS is technically supported, it is **not recommended**. Policy evaluation on macOS is also currently slow. We are working on adding a C-based MuJoCo backend, which will significantly speed up evaluation on macOS. Stay tuned for updates.

## Prerequisites

### Install uv

If you haven't already installed [uv](https://docs.astral.sh/uv/), run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation Methods

### Method 1: For mjlab Development (Recommended for Contributors)

If you're contributing to mjlab or need the latest changes:

1. Clone the repository:
```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
```

2. Run commands directly with uv (automatically creates venv and syncs dependencies):
```bash
uv run <your-command>
```

### Method 2: From Source (Recommended for Beta Users)

**Why source?** We're still in beta and actively fixing/improving things, so using the source version gives you the latest updates.

#### Option A: Local Editable Install

1. Clone the repository:
```bash
git clone https://github.com/mujocolab/mjlab.git
```

2. Add as an editable dependency to your project:
```bash
uv add --editable /path/to/cloned/mjlab
```

#### Option B: Direct Git Install

Install directly from GitHub without cloning:

```bash
uv add "mjlab @ git+https://github.com/mujocolab/mjlab" \
    "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp"
```

> **Note**: mujoco-warp must be installed from Git since it's not available on PyPI.

### Method 3: From PyPI (Stable Release)

For the stable release version:

```bash
uv add mjlab "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"
```

> **Note**: You still need to install mujoco-warp from Git even when using the PyPI version of mjlab.

## Verification

After installation, verify that mjlab is working by running the demo:

```bash
# If working inside the mjlab directory
uv run demo

# If mjlab is installed as a dependency in your project
uv run python -m mjlab.scripts.demo
```

## Troubleshooting

If you encounter any issues during installation or setup:
1. Check [faq.md](faq.md) for common questions and solutions
2. Open an issue on [GitHub Issues](https://github.com/mujocolab/mjlab/issues)
