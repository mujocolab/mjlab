"""Interactive diagnostics for D435 football detections."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from mjlab.scripts.sim2sim.d435_ball_observer import D435BallObserver, D435Config

FloatArray = npt.NDArray[np.float32]


class DetectionWindow:
  """Display YOLO image detections and robot-frame football vectors."""

  def __init__(self, cfg: D435Config, update_rate: float) -> None:
    if update_rate <= 0.0:
      raise ValueError("Detection window update rate must be positive.")

    # Importing pyplot selects a GUI backend, so defer it until the optional
    # interactive window is actually requested.
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    self._plt = plt
    self._update_period = 1.0 / update_rate
    self._last_update_time = -math.inf
    self._figure, (self._image_axis, self._vector_axis) = plt.subplots(
      1,
      2,
      figsize=(12.0, 5.4),
      num="D435 YOLO football detection",
    )
    self._image = self._image_axis.imshow(
      np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
    )
    self._box = Rectangle(
      (0.0, 0.0),
      0.0,
      0.0,
      fill=False,
      edgecolor="lime",
      linewidth=2.0,
      visible=False,
    )
    self._image_axis.add_patch(self._box)
    (self._bearing,) = self._image_axis.plot([], [], color="yellow", linewidth=2.0)
    self._status = self._image_axis.text(
      8,
      18,
      "Waiting for RGB frame...",
      color="white",
      fontsize=10,
      verticalalignment="top",
      bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )
    self._image_axis.set_title("D435 RGB + YOLO")
    self._image_axis.set_axis_off()
    self._principal_point = (0.5 * cfg.width, 0.5 * cfg.height)
    self._figure.tight_layout()
    plt.show(block=False)

  @property
  def is_open(self) -> bool:
    """Return whether the user has kept the diagnostic figure open."""
    return bool(self._plt.fignum_exists(self._figure.number))

  def update(
    self,
    observer: D435BallObserver,
    football_observation: tuple[FloatArray, FloatArray],
    sim_time: float,
  ) -> None:
    """Refresh the annotated camera image and top-down vector plot."""
    if not self.is_open or sim_time - self._last_update_time < self._update_period:
      return
    self._last_update_time = sim_time

    if observer.last_rgb is not None:
      self._image.set_data(observer.last_rgb)

    detection = observer.last_detection
    ball_position, feet_to_ball = football_observation
    if detection is None:
      self._box.set_visible(False)
      self._bearing.set_data([], [])
      state = "HELD" if np.any(np.abs(ball_position) > 1e-6) else "NOT DETECTED"
      self._status.set_text(f"YOLO: {state}")
    else:
      box, confidence = detection
      x1, y1, x2, y2 = (float(value) for value in box)
      center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
      self._box.set_xy((x1, y1))
      self._box.set_width(x2 - x1)
      self._box.set_height(y2 - y1)
      self._box.set_visible(True)
      self._bearing.set_data(
        (self._principal_point[0], center[0]),
        (self._principal_point[1], center[1]),
      )
      distance = float(np.linalg.norm(ball_position))
      bearing = math.degrees(
        math.atan2(float(ball_position[1]), float(ball_position[0]))
      )
      self._status.set_text(
        f"YOLO football  confidence={confidence:.3f}\n"
        f"bbox=({x1:.0f}, {y1:.0f})-({x2:.0f}, {y2:.0f})\n"
        f"ball_pos_b=({ball_position[0]:+.3f}, {ball_position[1]:+.3f}) m\n"
        f"distance={distance:.3f} m  bearing={bearing:+.1f} deg"
      )

    self._draw_vectors(ball_position, feet_to_ball)
    self._figure.canvas.draw_idle()
    self._figure.canvas.flush_events()

  def _draw_vectors(self, ball_position: FloatArray, feet_to_ball: FloatArray) -> None:
    axis = self._vector_axis
    axis.clear()
    axis.set_title("Robot yaw frame (top view)")
    axis.set_xlabel("x forward [m]")
    axis.set_ylabel("y left [m]")
    axis.grid(True, alpha=0.3)
    axis.set_aspect("equal", adjustable="box")

    ball = np.asarray(ball_position, dtype=np.float32)
    vectors = np.asarray(feet_to_ball, dtype=np.float32).reshape(2, 2)
    feet = ball[None, :] - vectors
    points = np.concatenate((np.zeros((1, 2)), ball[None, :], feet), axis=0)
    limit = max(0.5, float(np.max(np.abs(points))) * 1.25)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.scatter(0.0, 0.0, marker="^", s=80, color="deepskyblue", label="robot")

    if np.any(np.abs(ball) > 1e-6):
      axis.scatter(ball[0], ball[1], s=90, color="orange", label="YOLO ball")
      axis.arrow(
        0.0,
        0.0,
        float(ball[0]),
        float(ball[1]),
        color="orange",
        width=0.008,
        length_includes_head=True,
      )
      foot_colors = ("magenta", "limegreen")
      foot_labels = ("left foot", "right foot")
      for foot, vector, color, label in zip(
        feet, vectors, foot_colors, foot_labels, strict=True
      ):
        axis.scatter(foot[0], foot[1], s=45, color=color, label=label)
        axis.arrow(
          float(foot[0]),
          float(foot[1]),
          float(vector[0]),
          float(vector[1]),
          color=color,
          width=0.006,
          length_includes_head=True,
        )
      axis.text(
        0.02,
        0.02,
        "robot->ball "
        f"[{ball[0]:+.3f}, {ball[1]:+.3f}] m\n"
        f"left foot->ball  [{vectors[0, 0]:+.3f}, {vectors[0, 1]:+.3f}] m\n"
        f"right foot->ball [{vectors[1, 0]:+.3f}, {vectors[1, 1]:+.3f}] m",
        transform=axis.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 4},
      )
      axis.legend(loc="upper right", fontsize=8)

  def close(self) -> None:
    """Close the diagnostic figure if it is still open."""
    if self.is_open:
      self._plt.close(self._figure)
