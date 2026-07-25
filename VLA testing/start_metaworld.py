"""Launch a simple Meta-World pick-and-place environment."""

import time

import gymnasium as gym
import metaworld  # noqa: F401


def main() -> None:
  env = gym.make(
    "Meta-World/MT1",
    env_name="pick-place-v3",
    seed=42,
    render_mode="human",
  )
  observation, _ = env.reset()
  print(f"Meta-World started; observation shape: {observation.shape}")
  print("Running random actions. Close the viewer window to stop.")

  try:
    while True:
      env.render()
      observation, _, terminated, truncated, _ = env.step(
        env.action_space.sample()
      )
      if terminated or truncated:
        observation, _ = env.reset()
      time.sleep(1 / 60)
  except KeyboardInterrupt:
    print("Stopping Meta-World.")
  finally:
    env.close()


if __name__ == "__main__":
  main()
