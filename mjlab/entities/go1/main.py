import mujoco
from pathlib import Path

_HERE = Path(__file__).parent

model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(_HERE / "xmls" / "go1.xml"))
data = mujoco.MjData(model)

w_n = 30  # Hz.
damping_ratio = 1.0  # Critical damping.

stiffnesses = []
dampings = []
for i in range(model.nv):
  if i < 6:
    continue
  inertia = model.dof_M0[i]
  stiffness = inertia * w_n**2
  damping = 2 * damping_ratio * inertia * w_n
  stiffnesses.append(stiffness)
  dampings.append(damping)

for kp, kv in zip(stiffnesses, dampings):
  print(f"kp: {kp:.2f}, kv: {kv:.2f}")
