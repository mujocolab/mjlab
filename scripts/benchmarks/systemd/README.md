# Systemd Setup for Nightly Benchmarks

This directory contains systemd user service and timer files for running nightly
mjlab benchmarks.

## Setup

```bash
# 1. Create user systemd directory if it doesn't exist
mkdir -p ~/.config/systemd/user

# 2. Copy the service and timer files
cp mjlab-nightly.service ~/.config/systemd/user/
cp mjlab-nightly.timer ~/.config/systemd/user/

# 3. Edit the service file to set your WANDB_API_KEY
#    Get your key from: https://wandb.ai/authorize
nano ~/.config/systemd/user/mjlab-nightly.service

# 4. Reload systemd
systemctl --user daemon-reload

# 5. Enable and start the timer
systemctl --user enable mjlab-nightly.timer
systemctl --user start mjlab-nightly.timer

# 6. Enable lingering (so timer runs even when you're not logged in)
sudo loginctl enable-linger $USER
```

## Useful Commands

```bash
# Check timer status
systemctl --user status mjlab-nightly.timer

# List all timers and when they'll run next
systemctl --user list-timers

# Run the benchmark manually (without waiting for timer)
systemctl --user start mjlab-nightly.service

# View logs
journalctl --user -u mjlab-nightly.service -f

# View recent logs
journalctl --user -u mjlab-nightly.service --since "1 hour ago"

# Disable the timer
systemctl --user disable mjlab-nightly.timer
```

## Configuration

Edit `~/.config/systemd/user/mjlab-nightly.service` to customize:

- `Environment="CUDA_DEVICE=0"` - Which GPU to use
- `Environment="WANDB_API_KEY=..."` - Your WandB API key
- `MemoryMax=32G` - Memory limit for the training job

Edit `~/.config/systemd/user/mjlab-nightly.timer` to change the schedule:

- `OnCalendar=*-*-* 02:00:00` - Default: 2 AM daily
- `OnCalendar=Mon *-*-* 02:00:00` - Example: Mondays only at 2 AM
- `OnCalendar=*-*-* 02,14:00:00` - Example: Twice daily at 2 AM and 2 PM
