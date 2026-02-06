from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    super().save(path, infos)

  def load(
    self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
  ):
    infos = super().load(path, load_cfg, strict, map_location)
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
