from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroundContactCfg:
  """Configuration for ground contact solver parameters.

  These parameters control the softness/stiffness of ground contacts
  in MuJoCo simulation.

  Attributes:
      solref: Contact solver reference parameters [timeconst, dampratio].
          - timeconst: Time constant for contact dynamics (lower = stiffer).
          - dampratio: Damping ratio (1.0 = critically damped).
      solimp: Contact solver impedance parameters [dmin, dmax, width, midpoint, power].
          - dmin, dmax: Range of allowed impedance (0-1).
          - width: Transition zone width around contact.
          - midpoint: Midpoint of the transition.
          - power: Power of the transition curve.
  """

  solref: tuple[float, float] = (0.02, 1.0)
  """Contact reference: [time_constant, damping_ratio]."""
  solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
  """Contact impedance: [dmin, dmax, width, midpoint, power]."""
