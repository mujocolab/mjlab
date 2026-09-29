The eleven XML files are copied unchanged from `gymnasium==1.3.0`,
`gymnasium/envs/mujoco/assets/`.

Source: https://github.com/Farama-Foundation/Gymnasium/tree/v1.3.0/gymnasium/envs/mujoco/assets
License: MIT; see LICENSE. Original XML credits are retained.

These are complete benchmark scenes (robot, floor, target/object), so they remain
local to the task suite, following mjlab's Cartpole asset organization.
`asset_cfg.py` loads them with native `EntityCfg`, `EntityArticulationInfoCfg` and
`XmlActuatorCfg`. Setup adds stable names and force sensors in memory, preserves
the original motor definitions, and transfers global physics/compiler settings.
Gymnasium is not imported by the task runtime.
