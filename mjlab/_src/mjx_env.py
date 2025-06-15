from dataclasses import dataclass
from functools import partial
from typing import (
  Any,
  Callable,
  Dict,
  Generic,
  List,
  Mapping,
  Optional,
  Sequence,
  TypeVar,
  Union,
)

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import tqdm
from etils import epath
from mujoco import mjx

from mjlab._src import mjx_task
from mjlab._src.types import ObservationSize, State

TaskT = TypeVar("TaskT", bound=mjx_task.MjxTask)


@dataclass(frozen=True)
class MjxEnv(Generic[TaskT]):
  """Base class for playground environments."""

  task: TaskT

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    data = init(self.task.mjx_model)
    reward, done = jp.zeros(2)
    data, info, metrics = self.task.initialize_episode(data, rng)
    state = State(
      data=data,
      obs={},
      reward=reward,
      done=done,
      metrics=metrics,
      info=info,
    )
    obs = self.task.get_observation(data=data, state=state)
    return state.replace(obs=obs)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    data = self.task.before_step(action=action, state=state)
    data = step(
      model=self.task.mjx_model,
      data=data,
      n_substeps=self.task.n_substeps,
      before_substep_fn=partial(self.task.before_substep, action=action, state=state),
      after_substep_fn=partial(self.task.after_substep, action=action, state=state),
    )
    done = self.task.should_terminate_episode(data=data, state=state)
    obs = self.task.get_observation(data=data, state=state)
    reward = self.task.get_reward(data=data, state=state, action=action, done=done)
    data = self.task.after_step(data=data, state=state, action=action, done=done)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  @property
  def observation_size(self) -> ObservationSize:
    abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
    obs = abstract_state.obs
    if isinstance(obs, Mapping):
      return jax.tree_util.tree_map(lambda x: x.shape, obs)
    return obs.shape[-1]

  @property
  def action_size(self) -> int:
    return self.task.action_size

  def render(
    self,
    trajectory: List[State],
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
    scene_option: Optional[mujoco.MjvOption] = None,
    modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
  ) -> Sequence[np.ndarray]:
    return render_array(
      self.task.model,
      trajectory,
      height,
      width,
      camera,
      scene_option=scene_option,
      modify_scene_fns=modify_scene_fns,
    )

  @property
  def unwrapped(self) -> "MjxEnv[TaskT]":
    return self


def render_array(
  mj_model: mujoco.MjModel,
  trajectory: Union[List[State], State],
  height: int = 480,
  width: int = 640,
  camera: Optional[str] = None,
  scene_option: Optional[mujoco.MjvOption] = None,
  modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
  hfield_data: Optional[jax.Array] = None,
):
  """Renders a trajectory as an array of images."""
  renderer = mujoco.Renderer(mj_model, height=height, width=width)
  camera = camera or -1

  if hfield_data is not None:
    mj_model.hfield_data = hfield_data.reshape(mj_model.hfield_data.shape)
    mujoco.mjr_uploadHField(mj_model, renderer._mjr_context, 0)

  def get_image(state, modify_scn_fn=None) -> np.ndarray:
    d = mujoco.MjData(mj_model)
    d.qpos, d.qvel = state.data.qpos, state.data.qvel
    d.mocap_pos, d.mocap_quat = state.data.mocap_pos, state.data.mocap_quat
    d.xfrc_applied = state.data.xfrc_applied
    mujoco.mj_forward(mj_model, d)
    renderer.update_scene(d, camera=camera, scene_option=scene_option)
    if modify_scn_fn is not None:
      modify_scn_fn(renderer.scene)
    return renderer.render()

  if isinstance(trajectory, list):
    out = []
    for i, state in enumerate(tqdm.tqdm(trajectory)):
      if modify_scene_fns is not None:
        modify_scene_fn = modify_scene_fns[i]
      else:
        modify_scene_fn = None
      out.append(get_image(state, modify_scene_fn))
  else:
    out = get_image(trajectory)

  renderer.close()
  return out


def update_assets(
  assets: Dict[str, Any],
  path: Union[str, epath.Path],
  glob: str = "*",
  recursive: bool = False,
):
  """Update the assets dictionary with the contents of the given path.

  Args:
    assets: The dictionary to update.
    path: The path to the directory to update.
    glob: The glob pattern to use to find files.
    recursive: Whether to recursively update the assets dictionary.
  """
  for f in epath.Path(path).glob(glob):
    if f.is_file():
      assets[f.name] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, glob, recursive)


def init(model: mjx.Model) -> mjx.Data:
  """Initialize the physics state."""
  data = mjx.make_data(model)
  return mjx.forward(model, data)


def step(
  model: mjx.Model,
  data: mjx.Data,
  n_substeps: int = 1,
  before_substep_fn: Callable[[mjx.Data], mjx.Data] = lambda x: x,
  after_substep_fn: Callable[[mjx.Data], mjx.Data] = lambda x: x,
) -> mjx.Data:
  """Step the physics for the given number of substeps."""

  def substep_fn(_, data: mjx.Data) -> mjx.Data:
    data = before_substep_fn(data)
    data = mjx.step(model, data)
    data = after_substep_fn(data)
    return data

  return jax.lax.fori_loop(0, n_substeps, substep_fn, data)


def get_sensor_data(
  model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
  """Gets sensor data given sensor name."""
  sensor_id = model.sensor(sensor_name).id
  sensor_adr = model.sensor_adr[sensor_id]
  sensor_dim = model.sensor_dim[sensor_id]
  return data.sensordata[sensor_adr : sensor_adr + sensor_dim]
