from __future__ import annotations

__all__: list[str] = [
  "mjDISABLESTRING",
  "mjENABLESTRING",
  "mjFRAMESTRING",
  "mjLABELSTRING",
  "mjMAXCONPAIR",
  "mjMAXIMP",
  "mjMAXLIGHT",
  "mjMAXLINE",
  "mjMAXLINEPNT",
  "mjMAXOVERLAY",
  "mjMAXPLANEGRID",
  "mjMAXVAL",
  "mjMINIMP",
  "mjMINMU",
  "mjMINVAL",
  "mjNBIAS",
  "mjNDYN",
  "mjNEQDATA",
  "mjNGAIN",
  "mjNGROUP",
  "mjNIMP",
  "mjNREF",
  "mjNSOLVER",
  "mjPI",
  "mjRNDSTRING",
  "mjTIMERSTRING",
  "mjVERSION_HEADER",
  "mjVISSTRING",
]
mjDISABLESTRING: tuple = (
  "Constraint",
  "Equality",
  "Frictionloss",
  "Limit",
  "Contact",
  "Passive",
  "Gravity",
  "Clampctrl",
  "Warmstart",
  "Filterparent",
  "Actuation",
  "Refsafe",
  "Sensor",
  "Midphase",
  "Eulerdamp",
  "AutoReset",
  "NativeCCD",
)
mjENABLESTRING: tuple = (
  "Override",
  "Energy",
  "Fwdinv",
  "InvDiscrete",
  "MultiCCD",
  "Island",
)
mjFRAMESTRING: tuple = (
  "None",
  "Body",
  "Geom",
  "Site",
  "Camera",
  "Light",
  "Contact",
  "World",
)
mjLABELSTRING: tuple = (
  "None",
  "Body",
  "Joint",
  "Geom",
  "Site",
  "Camera",
  "Light",
  "Tendon",
  "Actuator",
  "Constraint",
  "Flex",
  "Skin",
  "Selection",
  "SelPoint",
  "Contact",
  "ContactForce",
  "Island",
)
mjMAXCONPAIR: int = 50
mjMAXIMP: float = 0.9999
mjMAXLIGHT: int = 100
mjMAXLINE: int = 100
mjMAXLINEPNT: int = 1001
mjMAXOVERLAY: int = 500
mjMAXPLANEGRID: int = 200
mjMAXVAL: float = 10000000000.0
mjMINIMP: float = 0.0001
mjMINMU: float = 1e-05
mjMINVAL: float = 1e-15
mjNBIAS: int = 10
mjNDYN: int = 10
mjNEQDATA: int = 11
mjNGAIN: int = 10
mjNGROUP: int = 6
mjNIMP: int = 5
mjNREF: int = 2
mjNSOLVER: int = 200
mjPI: float = 3.141592653589793
mjRNDSTRING: tuple = (
  ("Shadow", "1", "S"),
  ("Wireframe", "0", "W"),
  ("Reflection", "1", "R"),
  ("Additive", "0", "L"),
  ("Skybox", "1", "K"),
  ("Fog", "0", "G"),
  ("Haze", "1", "/"),
  ("Segment", "0", ","),
  ("Id Color", "0", ""),
  ("Cull Face", "1", ""),
)
mjTIMERSTRING: tuple = (
  "step",
  "forward",
  "inverse",
  "position",
  "velocity",
  "actuation",
  "constraint",
  "advance",
  "pos_kinematics",
  "pos_inertia",
  "pos_collision",
  "pos_make",
  "pos_project",
  "col_broadphase",
  "col_narrowphase",
)
mjVERSION_HEADER: int = 336
mjVISSTRING: tuple = (
  ("Convex Hull", "0", "H"),
  ("Texture", "1", "X"),
  ("Joint", "0", "J"),
  ("Camera", "0", "Q"),
  ("Actuator", "0", "U"),
  ("Activation", "0", ","),
  ("Light", "0", "Z"),
  ("Tendon", "1", "V"),
  ("Range Finder", "1", "Y"),
  ("Equality", "0", "E"),
  ("Inertia", "0", "I"),
  ("Scale Inertia", "0", "'"),
  ("Perturb Force", "0", "B"),
  ("Perturb Object", "1", "O"),
  ("Contact Point", "0", "C"),
  ("Island", "0", "N"),
  ("Contact Force", "0", "F"),
  ("Contact Split", "0", "P"),
  ("Transparent", "0", "T"),
  ("Auto Connect", "0", "A"),
  ("Center of Mass", "0", "M"),
  ("Select Point", "0", ""),
  ("Static Body", "1", "D"),
  ("Skin", "1", ";"),
  ("Flex Vert", "0", ""),
  ("Flex Edge", "1", ""),
  ("Flex Face", "0", ""),
  ("Flex Skin", "1", ""),
  ("Body Tree", "0", "`"),
  ("Mesh Tree", "0", "\\"),
  ("SDF iters", "0", ""),
)
