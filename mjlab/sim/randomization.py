from typing import Any, List
import warp as wp
import mujoco_warp as mjwarp


@wp.kernel(module="unique")
def repeat_array_kernel(
  src: wp.array(dtype=Any),
  nelems_per_world: int,
  dst: wp.array(dtype=Any),
):
  tid = wp.tid()
  src_idx = tid % nelems_per_world
  dst[tid] = src[src_idx]


def expand_model_fields(
  model: mjwarp.Model,
  nworld: int,
  fields_to_expand: List[str],
) -> None:
  if nworld == 1:
    return

  def tile(x: wp.array) -> wp.array:
    # Create new array with same shape but first dim multiplied by nworld
    new_shape = list(x.shape)
    new_shape[0] = nworld
    wp_array = {1: wp.array, 2: wp.array2d, 3: wp.array3d, 4: wp.array4d}[
      len(new_shape)
    ]
    dst = wp_array(shape=new_shape, dtype=x.dtype, device=x.device)

    # Flatten arrays for kernel
    src_flat = x.flatten()
    dst_flat = dst.flatten()

    # Launch kernel to repeat data - one thread per destination element
    n_elems_per_world = dst_flat.shape[0] // nworld
    wp.launch(
      repeat_array_kernel,
      dim=dst_flat.shape[0],
      inputs=[src_flat, n_elems_per_world],
      outputs=[dst_flat],
      device=x.device,
    )
    return dst

  for field in model.__dataclass_fields__:
    if field in fields_to_expand:
      array = getattr(model, field)
      setattr(model, field, tile(array))
