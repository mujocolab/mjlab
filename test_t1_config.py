"""Quick test to verify T1 velocity task configuration."""

from mjlab.tasks.velocity.config.t1 import booster_t1_flat_env_cfg

if __name__ == "__main__":
    cfg = booster_t1_flat_env_cfg()
    print("✓ T1 flat terrain config created successfully")
    print(f"  - Episode length: {cfg.episode_length_s}s")
    print(f"  - Robot: {list(cfg.scene.entities.keys())}")
    print(f"  - Sensors: {[s.name for s in cfg.scene.sensors] if cfg.scene.sensors else []}")
    print(f"  - Observations: {list(cfg.observations.keys())}")
    print(f"  - Rewards: {list(cfg.rewards.keys())}")
    print(f"  - Actions: {list(cfg.actions.keys())}")
    print("\n✓ All configurations loaded successfully!")
