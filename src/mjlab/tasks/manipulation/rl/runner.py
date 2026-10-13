from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner


class ManipulationOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper
  export_onnx_on_save = True
