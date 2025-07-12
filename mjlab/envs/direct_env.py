from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjwarp

from mjlab.entities.scene.scene import Scene
from mjlab.entities.scene.scene_config import SceneCfg, LightCfg
from mjlab.entities.common.config import TextureCfg
from mjlab.entities.robots.go1.go1_constants import GO1_ROBOT_CFG
from mjlab.entities.terrains.flat_terrain import FLAT_TERRAIN_CFG


SCENE_CFG = SceneCfg(
  terrains=(FLAT_TERRAIN_CFG,),
  robots=(GO1_ROBOT_CFG,),
  lights=(LightCfg(pos=(0, 0, 1.5), type="directional"),),
  skybox=TextureCfg(
    name="skybox",
    type="skybox",
    builtin="gradient",
    rgb1=(0.3, 0.5, 0.7),
    rgb2=(0.1, 0.2, 0.3),
    width=512,
    height=3072,
  ),
)

NUM_ENVS = 8
BATCH_SIZE = NUM_ENVS
NCONMAX = None
NJMAX = None
NSTEP = 1

scene = Scene(SCENE_CFG)
model: mujoco.MjModel = scene.compile()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
mujoco.mj_forward(model, data)

wp_model = mjwarp.put_model(model)
wp_data = mjwarp.put_data(model, data, nworld=BATCH_SIZE, nconmax=NCONMAX, njmax=NJMAX)

"""
1. sampling initial states and resetting to them.
2. extracting observations.
3. applying actions.
"""

t_qpos_init = torch.from_numpy(model.key(0).qpos).float()


def reset_to_keyframe(m: mjwarp.Model, d: mjwarp.Data):
  t_qpos = wp.from_torch(t_qpos_init)
  mjwarp.forward(m, d)


reset_to_keyframe(wp_model, wp_data)

# from ipdb import set_trace; set_trace()

device = torch.device("cuda:0")

with torch.no_grad():
  ctrl = torch.rand(size=(BATCH_SIZE, wp_model.nu), device=device)
print(f"ctrl: {ctrl.shape}")

wp_data.ctrl = wp.from_torch(ctrl)

mjwarp.step(wp_model, wp_data)

# jit_time, run_time, trace, ncon, nefc, solver_niter = mjwarp.benchmark(
#   mjwarp.step,
#   m=wp_model,
#   d=wp_data,
#   nstep=NSTEP,
#   event_trace=False,
#   measure_alloc=False,
#   measure_solver_niter=False,
# )
# steps = BATCH_SIZE * NSTEP

# print(f"""
# Summary for {BATCH_SIZE} parallel rollouts
# # Total JIT time: {jit_time:.2f} s
# # Total simulation time: {run_time:.2f} s
# # Total steps per second: {steps / run_time:,.0f}
# # Total realtime factor: {steps * wp_model.opt.timestep.numpy()[0] / run_time:,.2f} x
# # Total time per step: {1e9 * run_time / steps:.2f} ns"""
# )
