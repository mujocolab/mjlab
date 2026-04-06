from __future__ import annotations
import collections.abc
import mujoco._enums
import mujoco._structs
import numpy
import numpy.typing
import typing
__all__: list[str] = ['MjByteVec', 'MjCharVec', 'MjDoubleVec', 'MjFloatVec', 'MjIntVec', 'MjOption', 'MjSpec', 'MjStatistic', 'MjStringVec', 'MjVisual', 'MjVisualHeadlight', 'MjVisualRgba', 'MjsActuator', 'MjsBody', 'MjsCamera', 'MjsCompiler', 'MjsDefault', 'MjsElement', 'MjsEquality', 'MjsExclude', 'MjsFlex', 'MjsFrame', 'MjsGeom', 'MjsHField', 'MjsJoint', 'MjsKey', 'MjsLight', 'MjsMaterial', 'MjsMesh', 'MjsNumeric', 'MjsOrientation', 'MjsPair', 'MjsPlugin', 'MjsSensor', 'MjsSite', 'MjsSkin', 'MjsTendon', 'MjsTendonPath', 'MjsText', 'MjsTexture', 'MjsTuple', 'MjsWrap']
class MjByteVec:
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> ...:
        ...
    def __init__(self, arg0: ..., arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[...]:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: ...) -> None:
        ...
class MjCharVec:
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
        ...
    def __init__(self, arg0: str, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[str]:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: str) -> None:
        ...
class MjDoubleVec:
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def __init__(self, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[float]:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MjFloatVec:
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def __init__(self, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[float]:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MjIntVec:
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> int:
        ...
    def __init__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[int]:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class MjOption:
    @property
    def ccd_iterations(self) -> int:
        ...
    @ccd_iterations.setter
    def ccd_iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ccd_tolerance(self) -> float:
        ...
    @ccd_tolerance.setter
    def ccd_tolerance(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cone(self) -> int:
        ...
    @cone.setter
    def cone(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def density(self) -> float:
        ...
    @density.setter
    def density(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def disableactuator(self) -> int:
        ...
    @disableactuator.setter
    def disableactuator(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def disableflags(self) -> int:
        ...
    @disableflags.setter
    def disableflags(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def enableflags(self) -> int:
        ...
    @enableflags.setter
    def enableflags(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def gravity(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @gravity.setter
    def gravity(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def impratio(self) -> float:
        ...
    @impratio.setter
    def impratio(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def integrator(self) -> int:
        ...
    @integrator.setter
    def integrator(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def iterations(self) -> int:
        ...
    @iterations.setter
    def iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def jacobian(self) -> int:
        ...
    @jacobian.setter
    def jacobian(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ls_iterations(self) -> int:
        ...
    @ls_iterations.setter
    def ls_iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ls_tolerance(self) -> float:
        ...
    @ls_tolerance.setter
    def ls_tolerance(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def magnetic(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @magnetic.setter
    def magnetic(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def noslip_iterations(self) -> int:
        ...
    @noslip_iterations.setter
    def noslip_iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def noslip_tolerance(self) -> float:
        ...
    @noslip_tolerance.setter
    def noslip_tolerance(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o_friction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @o_friction.setter
    def o_friction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def o_margin(self) -> float:
        ...
    @o_margin.setter
    def o_margin(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o_solimp(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @o_solimp.setter
    def o_solimp(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def o_solref(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @o_solref.setter
    def o_solref(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def sdf_initpoints(self) -> int:
        ...
    @sdf_initpoints.setter
    def sdf_initpoints(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def sdf_iterations(self) -> int:
        ...
    @sdf_iterations.setter
    def sdf_iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def sleep_tolerance(self) -> float:
        ...
    @sleep_tolerance.setter
    def sleep_tolerance(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def solver(self) -> int:
        ...
    @solver.setter
    def solver(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def timestep(self) -> float:
        ...
    @timestep.setter
    def timestep(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tolerance(self) -> float:
        ...
    @tolerance.setter
    def tolerance(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def viscosity(self) -> float:
        ...
    @viscosity.setter
    def viscosity(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wind(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @wind.setter
    def wind(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
class MjSpec:
    assets: dict
    comment: str
    compiler: MjsCompiler
    meshdir: str
    modelfiledir: str
    modelname: str
    option: MjOption
    override_assets: bool
    stat: MjStatistic
    texturedir: str
    visual: MjVisual
    @staticmethod
    def from_file(filename: str, include: collections.abc.Mapping[str, bytes] | None = None, assets: dict | None = None) -> MjSpec:
        """
            Creates a spec from an XML file.
        
            Parameters
            ----------
            filename : str
                Path to the XML file.
            include : dict, optional
                A dictionary of xml files included by the model. The keys are file names
                and the values are file contents.
            assets : dict, optional
                A dictionary of assets to be used by the spec. The keys are asset names
                and the values are asset contents.
        """
    @staticmethod
    def from_string(xml: str, include: collections.abc.Mapping[str, bytes] | None = None, assets: dict | None = None) -> MjSpec:
        """
            Creates a spec from an XML string.
        
            Parameters
            ----------
            xml : str
                XML string.
            include : dict, optional
                A dictionary of xml files included by the model. The keys are file names
                and the values are file contents.
            assets : dict, optional
                A dictionary of assets to be used by the spec. The keys are asset names
                and the values are asset contents.
        """
    @staticmethod
    def from_zip(file: typing.Union[str, typing.IO[bytes]]) -> MjSpec:
        """
        Reads a zip file and returns an MjSpec.
        
        Args:
          file: The path to the file to read from or the file object to read from.
        Returns:
          An MjSpec object.
        """
    @staticmethod
    def resolve_orientation(degree: bool, sequence: MjCharVec = None, orientation: MjsOrientation) -> typing.Annotated[list[float], "FixedSize(4)"]:
        ...
    @staticmethod
    def to_zip(spec: MjSpec, file: typing.Union[str, typing.IO[bytes]]) -> None:
        """
        Converts an MjSpec to a zip file.
        
        Args:
          spec: The mjSpec to save to a file.
          file: The path to the file to save to or the file object to write to.
        """
    def __init__(self) -> None:
        ...
    def activate_plugin(self, name: str) -> None:
        ...
    def actuator(self, arg0: str) -> MjsActuator:
        ...
    def add_actuator(self, default: MjsDefault = None, name: str | None = None, gaintype: typing.SupportsInt | typing.SupportsIndex | None = None, gainprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, biastype: typing.SupportsInt | typing.SupportsIndex | None = None, biasprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, dyntype: typing.SupportsInt | typing.SupportsIndex | None = None, dynprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actdim: typing.SupportsInt | typing.SupportsIndex | None = None, actearly: typing.SupportsInt | typing.SupportsIndex | None = None, trntype: typing.SupportsInt | typing.SupportsIndex | None = None, gear: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, target: str | None = None, refsite: str | None = None, slidersite: str | None = None, cranklength: typing.SupportsFloat | typing.SupportsIndex | None = None, lengthrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inheritrange: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: typing.Any | None = None, armature: typing.SupportsFloat | typing.SupportsIndex | None = None, ctrllimited: typing.SupportsInt | typing.SupportsIndex | None = None, ctrlrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, forcelimited: typing.SupportsInt | typing.SupportsIndex | None = None, forcerange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actlimited: typing.SupportsInt | typing.SupportsIndex | None = None, actrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, nsample: typing.SupportsInt | typing.SupportsIndex | None = None, interp: typing.SupportsInt | typing.SupportsIndex | None = None, delay: typing.SupportsFloat | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> MjsActuator:
        """
              Add actuator to spec.
        
              Args:
                name: str
                gaintype: int
                gainprm: list[float]
                biastype: int
                biasprm: list[float]
                dyntype: int
                dynprm: list[float]
                actdim: int
                actearly: int
                trntype: int
                gear: list[float]
                target: str
                refsite: str
                slidersite: str
                cranklength: float
                lengthrange: list[float]
                inheritrange: float
                damping: Optional[list[float]]
                armature: float
                ctrllimited: int
                ctrlrange: list[float]
                forcelimited: int
                forcerange: list[float]
                actlimited: int
                actrange: list[float]
                group: int
                nsample: int
                interp: int
                delay: float
                userdata: list[float]
                plugin: MjsPlugin
                info: str
        """
    def add_default(self, arg0: str, arg1: MjsDefault) -> MjsDefault:
        ...
    def add_equality(self, default: MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, active: typing.SupportsInt | typing.SupportsIndex | None = None, name1: str | None = None, name2: str | None = None, objtype: typing.SupportsInt | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsEquality:
        """
              Add equality to spec.
        
              Args:
                name: str
                type: int
                data: list[float]
                active: int
                name1: str
                name2: str
                objtype: int
                solref: list[float]
                solimp: list[float]
                info: str
        """
    def add_exclude(self, name: str | None = None, bodyname1: str | None = None, bodyname2: str | None = None, info: str | None = None) -> MjsExclude:
        """
              Add exclude to spec.
        
              Args:
                name: str
                bodyname1: str
                bodyname2: str
                info: str
        """
    def add_flex(self, name: str | None = None, contype: typing.SupportsInt | typing.SupportsIndex | None = None, conaffinity: typing.SupportsInt | typing.SupportsIndex | None = None, condim: typing.SupportsInt | typing.SupportsIndex | None = None, priority: typing.SupportsInt | typing.SupportsIndex | None = None, friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solmix: typing.SupportsFloat | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, gap: typing.SupportsFloat | typing.SupportsIndex | None = None, dim: typing.SupportsInt | typing.SupportsIndex | None = None, radius: typing.SupportsFloat | typing.SupportsIndex | None = None, size: typing.Any | None = None, internal: typing.SupportsInt | typing.SupportsIndex | None = None, flatskin: typing.SupportsInt | typing.SupportsIndex | None = None, selfcollide: typing.SupportsInt | typing.SupportsIndex | None = None, passive: typing.SupportsInt | typing.SupportsIndex | None = None, activelayers: typing.SupportsInt | typing.SupportsIndex | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, edgestiffness: typing.SupportsFloat | typing.SupportsIndex | None = None, edgedamping: typing.SupportsFloat | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, material: str | None = None, young: typing.SupportsFloat | typing.SupportsIndex | None = None, poisson: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: typing.SupportsFloat | typing.SupportsIndex | None = None, thickness: typing.SupportsFloat | typing.SupportsIndex | None = None, elastic2d: typing.SupportsInt | typing.SupportsIndex | None = None, nodebody: collections.abc.Sequence[str] | None = None, vertbody: collections.abc.Sequence[str] | None = None, node: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, vert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, elem: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, texcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, elemtexcoord: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsFlex:
        """
              Add flex to spec.
        
              Args:
                name: str
                contype: int
                conaffinity: int
                condim: int
                priority: int
                friction: list[float]
                solmix: float
                solref: list[float]
                solimp: list[float]
                margin: float
                gap: float
                dim: int
                radius: float
                size: Optional[list[float]]
                internal: int
                flatskin: int
                selfcollide: int
                passive: int
                activelayers: int
                group: int
                edgestiffness: float
                edgedamping: float
                rgba: list[float]
                material: str
                young: float
                poisson: float
                damping: float
                thickness: float
                elastic2d: int
                nodebody: list[str]
                vertbody: list[str]
                node: list[float]
                vert: list[float]
                elem: list[int]
                texcoord: list[float]
                elemtexcoord: list[int]
                info: str
        """
    def add_hfield(self, name: str | None = None, content_type: str | None = None, file: str | None = None, size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, nrow: typing.SupportsInt | typing.SupportsIndex | None = None, ncol: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsHField:
        """
              Add hfield to spec.
        
              Args:
                name: str
                content_type: str
                file: str
                size: list[float]
                nrow: int
                ncol: int
                userdata: list[float]
                info: str
        """
    def add_key(self, name: str | None = None, time: typing.SupportsFloat | typing.SupportsIndex | None = None, qpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, qvel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, act: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ctrl: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsKey:
        """
              Add key to spec.
        
              Args:
                name: str
                time: float
                qpos: list[float]
                qvel: list[float]
                act: list[float]
                mpos: list[float]
                mquat: list[float]
                ctrl: list[float]
                info: str
        """
    def add_material(self, default: MjsDefault = None, name: str | None = None, textures: collections.abc.Sequence[str] | None = None, texuniform: typing.SupportsInt | typing.SupportsIndex | None = None, texrepeat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, emission: typing.SupportsFloat | typing.SupportsIndex | None = None, specular: typing.SupportsFloat | typing.SupportsIndex | None = None, shininess: typing.SupportsFloat | typing.SupportsIndex | None = None, reflectance: typing.SupportsFloat | typing.SupportsIndex | None = None, metallic: typing.SupportsFloat | typing.SupportsIndex | None = None, roughness: typing.SupportsFloat | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsMaterial:
        """
              Add material to spec.
        
              Args:
                name: str
                textures: list[str]
                texuniform: int
                texrepeat: list[float]
                emission: float
                specular: float
                shininess: float
                reflectance: float
                metallic: float
                roughness: float
                rgba: list[float]
                info: str
        """
    def add_mesh(self, default: MjsDefault = None, name: str | None = None, content_type: str | None = None, file: str | None = None, refpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, refquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, scale: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inertia: typing.SupportsInt | typing.SupportsIndex | None = None, smoothnormal: typing.SupportsInt | typing.SupportsIndex | None = None, needsdf: typing.SupportsInt | typing.SupportsIndex | None = None, maxhullvert: typing.SupportsInt | typing.SupportsIndex | None = None, uservert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, usernormal: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, usertexcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userface: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, userfacenormal: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, userfacetexcoord: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, material: str | None = None, info: str | None = None) -> MjsMesh:
        """
              Add mesh to spec.
        
              Args:
                name: str
                content_type: str
                file: str
                refpos: list[float]
                refquat: list[float]
                scale: list[float]
                inertia: int
                smoothnormal: int
                needsdf: int
                maxhullvert: int
                uservert: list[float]
                usernormal: list[float]
                usertexcoord: list[float]
                userface: list[int]
                userfacenormal: list[int]
                userfacetexcoord: list[int]
                plugin: MjsPlugin
                material: str
                info: str
        """
    def add_numeric(self, name: str | None = None, data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: typing.SupportsInt | typing.SupportsIndex | None = None, info: str | None = None) -> MjsNumeric:
        """
              Add numeric to spec.
        
              Args:
                name: str
                data: list[float]
                size: int
                info: str
        """
    def add_pair(self, default: MjsDefault = None, name: str | None = None, geomname1: str | None = None, geomname2: str | None = None, condim: typing.SupportsInt | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solreffriction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, gap: typing.SupportsFloat | typing.SupportsIndex | None = None, friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsPair:
        """
              Add pair to spec.
        
              Args:
                name: str
                geomname1: str
                geomname2: str
                condim: int
                solref: list[float]
                solreffriction: list[float]
                solimp: list[float]
                margin: float
                gap: float
                friction: list[float]
                info: str
        """
    def add_plugin(self, name: str | None = None, plugin_name: str | None = None, active: typing.SupportsInt | typing.SupportsIndex | None = None, info: str | None = None) -> MjsPlugin:
        """
              Add plugin to spec.
        
              Args:
                name: str
                plugin_name: str
                active: int
                info: str
        """
    def add_sensor(self, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, objtype: typing.SupportsInt | typing.SupportsIndex | None = None, objname: str | None = None, reftype: typing.SupportsInt | typing.SupportsIndex | None = None, refname: str | None = None, intprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, datatype: typing.SupportsInt | typing.SupportsIndex | None = None, needstage: typing.SupportsInt | typing.SupportsIndex | None = None, dim: typing.SupportsInt | typing.SupportsIndex | None = None, cutoff: typing.SupportsFloat | typing.SupportsIndex | None = None, noise: typing.SupportsFloat | typing.SupportsIndex | None = None, nsample: typing.SupportsInt | typing.SupportsIndex | None = None, interp: typing.SupportsInt | typing.SupportsIndex | None = None, delay: typing.SupportsFloat | typing.SupportsIndex | None = None, interval: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> MjsSensor:
        """
              Add sensor to spec.
        
              Args:
                name: str
                type: int
                objtype: int
                objname: str
                reftype: int
                refname: str
                intprm: list[float]
                datatype: int
                needstage: int
                dim: int
                cutoff: float
                noise: float
                nsample: int
                interp: int
                delay: float
                interval: list[float]
                userdata: list[float]
                plugin: MjsPlugin
                info: str
        """
    def add_skin(self, name: str | None = None, file: str | None = None, material: str | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inflate: typing.SupportsFloat | typing.SupportsIndex | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, vert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, texcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, face: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, bodyname: collections.abc.Sequence[str] | None = None, bindpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, bindquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, vertid: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]] | None = None, vertweight: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]] | None = None, info: str | None = None) -> MjsSkin:
        """
              Add skin to spec.
        
              Args:
                name: str
                file: str
                material: str
                rgba: list[float]
                inflate: float
                group: int
                vert: list[float]
                texcoord: list[float]
                face: list[int]
                bodyname: list[str]
                bindpos: list[float]
                bindquat: list[float]
                vertid: list[list[int]]
                vertweight: list[list[float]]
                info: str
        """
    def add_tendon(self, default: MjsDefault = None, name: str | None = None, stiffness: typing.Any | None = None, springlength: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, damping: typing.Any | None = None, frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, armature: typing.SupportsFloat | typing.SupportsIndex | None = None, limited: typing.SupportsInt | typing.SupportsIndex | None = None, actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = None, range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, material: str | None = None, width: typing.SupportsFloat | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsTendon:
        """
              Add tendon to spec.
        
              Args:
                name: str
                stiffness: Optional[list[float]]
                springlength: list[float]
                damping: Optional[list[float]]
                frictionloss: float
                solref_friction: list[float]
                solimp_friction: list[float]
                armature: float
                limited: int
                actfrclimited: int
                range: list[float]
                actfrcrange: list[float]
                margin: float
                solref_limit: list[float]
                solimp_limit: list[float]
                material: str
                width: float
                rgba: list[float]
                group: int
                userdata: list[float]
                info: str
        """
    def add_text(self, name: str | None = None, data: str | None = None, info: str | None = None) -> MjsText:
        """
              Add text to spec.
        
              Args:
                name: str
                data: str
                info: str
        """
    def add_texture(self, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, colorspace: typing.SupportsInt | typing.SupportsIndex | None = None, builtin: typing.SupportsInt | typing.SupportsIndex | None = None, mark: typing.SupportsInt | typing.SupportsIndex | None = None, rgb1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, rgb2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, markrgb: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, random: typing.SupportsFloat | typing.SupportsIndex | None = None, height: typing.SupportsInt | typing.SupportsIndex | None = None, width: typing.SupportsInt | typing.SupportsIndex | None = None, nchannel: typing.SupportsInt | typing.SupportsIndex | None = None, content_type: str | None = None, file: str | None = None, gridsize: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, gridlayout: typing.Any = None, cubefiles: collections.abc.Sequence[str] | None = None, data: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, hflip: typing.SupportsInt | typing.SupportsIndex | None = None, vflip: typing.SupportsInt | typing.SupportsIndex | None = None, info: str | None = None) -> MjsTexture:
        """
              Add texture to spec.
        
              Args:
                name: str
                type: int
                colorspace: int
                builtin: int
                mark: int
                rgb1: list[float]
                rgb2: list[float]
                markrgb: list[float]
                random: float
                height: int
                width: int
                nchannel: int
                content_type: str
                file: str
                gridsize: list[float]
                gridlayout: str | list[str]
                cubefiles: list[str]
                data: list[int]
                hflip: int
                vflip: int
                info: str
        """
    def add_tuple(self, name: str | None = None, objtype: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, objname: collections.abc.Sequence[str] | None = None, objprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsTuple:
        """
              Add tuple to spec.
        
              Args:
                name: str
                objtype: list[int]
                objname: list[str]
                objprm: list[float]
                info: str
        """
    def attach(self, child: MjSpec, prefix: str | None = None, suffix: str | None = None, site: typing.Any | None = None, frame: typing.Any | None = None) -> MjsFrame:
        ...
    def body(self, arg0: str) -> MjsBody:
        ...
    def camera(self, arg0: str) -> MjsCamera:
        ...
    def compile(self) -> typing.Any:
        ...
    def copy(self) -> MjSpec:
        ...
    @typing.overload
    def delete(self, arg0: MjsDefault) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsBody) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsFrame) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsGeom) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsJoint) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsSite) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsCamera) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsLight) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsMaterial) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsMesh) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsPair) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsEquality) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsActuator) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsTendon) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsSensor) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsFlex) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsHField) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsSkin) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsTexture) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsKey) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsText) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsNumeric) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsExclude) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsTuple) -> None:
        ...
    @typing.overload
    def delete(self, arg0: MjsPlugin) -> None:
        ...
    def equality(self, arg0: str) -> MjsEquality:
        ...
    def exclude(self, arg0: str) -> MjsExclude:
        ...
    def find_default(self, arg0: str) -> MjsDefault:
        ...
    def flex(self, arg0: str) -> MjsFlex:
        ...
    def frame(self, arg0: str) -> MjsFrame:
        ...
    def geom(self, arg0: str) -> MjsGeom:
        ...
    def hfield(self, arg0: str) -> MjsHField:
        ...
    def joint(self, arg0: str) -> MjsJoint:
        ...
    def key(self, arg0: str) -> MjsKey:
        ...
    def light(self, arg0: str) -> MjsLight:
        ...
    def material(self, arg0: str) -> MjsMaterial:
        ...
    def mesh(self, arg0: str) -> MjsMesh:
        ...
    def numeric(self, arg0: str) -> MjsNumeric:
        ...
    def pair(self, arg0: str) -> MjsPair:
        ...
    def plugin(self, arg0: str) -> MjsPlugin:
        ...
    def recompile(self, arg0: typing.Any, arg1: typing.Any) -> typing.Any:
        ...
    def sensor(self, arg0: str) -> MjsSensor:
        ...
    def site(self, arg0: str) -> MjsSite:
        ...
    def skin(self, arg0: str) -> MjsSkin:
        ...
    def tendon(self, arg0: str) -> MjsTendon:
        ...
    def text(self, arg0: str) -> MjsText:
        ...
    def texture(self, arg0: str) -> MjsTexture:
        ...
    def to_file(self, arg0: str) -> None:
        ...
    def to_xml(self) -> str:
        ...
    def tuple(self, arg0: str) -> MjsTuple:
        ...
    @property
    def _address(self) -> int:
        ...
    @property
    def actuators(self) -> list:
        ...
    @property
    def bodies(self) -> list:
        ...
    @property
    def cameras(self) -> list:
        ...
    @property
    def copy_during_attach(self) -> None:
        ...
    @copy_during_attach.setter
    def copy_during_attach(self, arg1: bool) -> int:
        ...
    @property
    def default(self) -> MjsDefault:
        ...
    @property
    def equalities(self) -> list:
        ...
    @property
    def excludes(self) -> list:
        ...
    @property
    def flexes(self) -> list:
        ...
    @property
    def frames(self) -> list:
        ...
    @property
    def geoms(self) -> list:
        ...
    @property
    def hasImplicitPluginElem(self) -> int:
        ...
    @hasImplicitPluginElem.setter
    def hasImplicitPluginElem(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def hfields(self) -> list:
        ...
    @property
    def joints(self) -> list:
        ...
    @property
    def keys(self) -> list:
        ...
    @property
    def lights(self) -> list:
        ...
    @property
    def materials(self) -> list:
        ...
    @property
    def memory(self) -> int:
        ...
    @memory.setter
    def memory(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def meshes(self) -> list:
        ...
    @property
    def nconmax(self) -> int:
        ...
    @nconmax.setter
    def nconmax(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nemax(self) -> int:
        ...
    @nemax.setter
    def nemax(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def njmax(self) -> int:
        ...
    @njmax.setter
    def njmax(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nkey(self) -> int:
        ...
    @nkey.setter
    def nkey(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nstack(self) -> int:
        ...
    @nstack.setter
    def nstack(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def numerics(self) -> list:
        ...
    @property
    def nuser_actuator(self) -> int:
        ...
    @nuser_actuator.setter
    def nuser_actuator(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nuser_body(self) -> int:
        ...
    @nuser_body.setter
    def nuser_body(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nuser_cam(self) -> int:
        ...
    @nuser_cam.setter
    def nuser_cam(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nuser_geom(self) -> int:
        ...
    @nuser_geom.setter
    def nuser_geom(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nuser_jnt(self) -> int:
        ...
    @nuser_jnt.setter
    def nuser_jnt(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nuser_sensor(self) -> int:
        ...
    @nuser_sensor.setter
    def nuser_sensor(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nuser_site(self) -> int:
        ...
    @nuser_site.setter
    def nuser_site(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nuser_tendon(self) -> int:
        ...
    @nuser_tendon.setter
    def nuser_tendon(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nuserdata(self) -> int:
        ...
    @nuserdata.setter
    def nuserdata(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def pairs(self) -> list:
        ...
    @property
    def parent(self) -> MjSpec:
        ...
    @property
    def plugins(self) -> list:
        ...
    @property
    def sensors(self) -> list:
        ...
    @property
    def sites(self) -> list:
        ...
    @property
    def skins(self) -> list:
        ...
    @property
    def strippath(self) -> int:
        ...
    @strippath.setter
    def strippath(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def tendons(self) -> list:
        ...
    @property
    def texts(self) -> list:
        ...
    @property
    def textures(self) -> list:
        ...
    @property
    def tuples(self) -> list:
        ...
    @property
    def worldbody(self) -> MjsBody:
        ...
class MjStatistic:
    @property
    def center(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @center.setter
    def center(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def extent(self) -> float:
        ...
    @extent.setter
    def extent(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def meaninertia(self) -> float:
        ...
    @meaninertia.setter
    def meaninertia(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def meanmass(self) -> float:
        ...
    @meanmass.setter
    def meanmass(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def meansize(self) -> float:
        ...
    @meansize.setter
    def meansize(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MjStringVec:
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
        ...
    def __init__(self, arg0: str, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[str]:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: str) -> None:
        ...
class MjVisual:
    global: mujoco._structs.MjVisual.Global
    global_: mujoco._structs.MjVisual.Global
    headlight: MjVisualHeadlight
    map: mujoco._structs.MjVisual.Map
    quality: mujoco._structs.MjVisual.Quality
    rgba: MjVisualRgba
    scale: mujoco._structs.MjVisual.Scale
class MjVisualHeadlight:
    @property
    def active(self) -> int:
        ...
    @active.setter
    def active(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ambient(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]", "flags.writeable"]:
        ...
    @ambient.setter
    def ambient(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]) -> None:
        ...
    @property
    def diffuse(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]", "flags.writeable"]:
        ...
    @diffuse.setter
    def diffuse(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]) -> None:
        ...
    @property
    def specular(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]", "flags.writeable"]:
        ...
    @specular.setter
    def specular(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]) -> None:
        ...
class MjVisualRgba:
    @property
    def actuator(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @actuator.setter
    def actuator(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def actuatornegative(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @actuatornegative.setter
    def actuatornegative(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def actuatorpositive(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @actuatorpositive.setter
    def actuatorpositive(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def bv(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @bv.setter
    def bv(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def bvactive(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @bvactive.setter
    def bvactive(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def camera(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @camera.setter
    def camera(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def com(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @com.setter
    def com(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def connect(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @connect.setter
    def connect(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def constraint(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @constraint.setter
    def constraint(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def contactforce(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @contactforce.setter
    def contactforce(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def contactfriction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @contactfriction.setter
    def contactfriction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def contactgap(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @contactgap.setter
    def contactgap(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def contactpoint(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @contactpoint.setter
    def contactpoint(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def contacttorque(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @contacttorque.setter
    def contacttorque(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def crankbroken(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @crankbroken.setter
    def crankbroken(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def fog(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @fog.setter
    def fog(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def force(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @force.setter
    def force(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def frustum(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @frustum.setter
    def frustum(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def haze(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @haze.setter
    def haze(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def inertia(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @inertia.setter
    def inertia(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def joint(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @joint.setter
    def joint(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def light(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @light.setter
    def light(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def rangefinder(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @rangefinder.setter
    def rangefinder(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def selectpoint(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @selectpoint.setter
    def selectpoint(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def slidercrank(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @slidercrank.setter
    def slidercrank(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
class MjsActuator:
    biastype: mujoco._enums.mjtBias
    classname: MjsDefault
    dyntype: mujoco._enums.mjtDyn
    gaintype: mujoco._enums.mjtGain
    info: str
    name: str
    plugin: MjsPlugin
    refsite: str
    slidersite: str
    target: str
    trntype: mujoco._enums.mjtTrn
    def set_to_adhesion(self, gain: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_to_cylinder(self, timeconst: typing.SupportsFloat | typing.SupportsIndex, bias: typing.SupportsFloat | typing.SupportsIndex, area: typing.SupportsFloat | typing.SupportsIndex, diameter: typing.SupportsFloat | typing.SupportsIndex = -1) -> None:
        ...
    def set_to_damper(self, kv: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_to_dcmotor(self, motorconst: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"], resistance: typing.SupportsFloat | typing.SupportsIndex, nominal: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 0.0], saturation: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(4)"] = [0.0, 0.0, 0.0, 0.0], inductance: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"] = [0.0, 0.0], cogging: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 0.0], controller: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(5)"] = [0.0, 0.0, 0.0, 0.0, 0.0], thermal: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(6)"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lugre: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(6)"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], input_mode: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def set_to_intvelocity(self, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = -1, dampratio: typing.SupportsFloat | typing.SupportsIndex = -1, timeconst: typing.SupportsFloat | typing.SupportsIndex = -1, inheritrange: bool = False) -> None:
        ...
    def set_to_motor(self) -> None:
        ...
    def set_to_muscle(self, timeconst: typing.SupportsFloat | typing.SupportsIndex = -1, tausmooth: typing.SupportsFloat | typing.SupportsIndex, range: typing.SupportsFloat | typing.SupportsIndex = [-1.0, -1.0], force: typing.SupportsFloat | typing.SupportsIndex = -1, scale: typing.SupportsFloat | typing.SupportsIndex = -1, lmin: typing.SupportsFloat | typing.SupportsIndex = -1, lmax: typing.SupportsFloat | typing.SupportsIndex = -1, vmax: typing.SupportsFloat | typing.SupportsIndex = -1, fpmax: typing.SupportsFloat | typing.SupportsIndex = -1, fvmax: typing.SupportsFloat | typing.SupportsIndex = -1) -> None:
        ...
    def set_to_position(self, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = -1, dampratio: typing.SupportsFloat | typing.SupportsIndex = -1, timeconst: typing.SupportsFloat | typing.SupportsIndex = -1, inheritrange: bool = False) -> None:
        ...
    def set_to_velocity(self, kv: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def actdim(self) -> int:
        ...
    @actdim.setter
    def actdim(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def actearly(self) -> int:
        ...
    @actearly.setter
    def actearly(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def actlimited(self) -> int:
        ...
    @actlimited.setter
    def actlimited(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def actrange(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @actrange.setter
    def actrange(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def armature(self) -> float:
        ...
    @armature.setter
    def armature(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def biasprm(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[10, 1]", "flags.writeable"]:
        ...
    @biasprm.setter
    def biasprm(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[10, 1]"]) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def cranklength(self) -> float:
        ...
    @cranklength.setter
    def cranklength(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ctrllimited(self) -> int:
        ...
    @ctrllimited.setter
    def ctrllimited(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ctrlrange(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @ctrlrange.setter
    def ctrlrange(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def damping(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @damping.setter
    def damping(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def delay(self) -> float:
        ...
    @delay.setter
    def delay(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dynprm(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[10, 1]", "flags.writeable"]:
        ...
    @dynprm.setter
    def dynprm(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[10, 1]"]) -> None:
        ...
    @property
    def forcelimited(self) -> int:
        ...
    @forcelimited.setter
    def forcelimited(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def forcerange(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @forcerange.setter
    def forcerange(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def gainprm(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[10, 1]", "flags.writeable"]:
        ...
    @gainprm.setter
    def gainprm(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[10, 1]"]) -> None:
        ...
    @property
    def gear(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"]:
        ...
    @gear.setter
    def gear(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]"]) -> None:
        ...
    @property
    def group(self) -> int:
        ...
    @group.setter
    def group(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def inheritrange(self) -> float:
        ...
    @inheritrange.setter
    def inheritrange(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def interp(self) -> int:
        ...
    @interp.setter
    def interp(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def lengthrange(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @lengthrange.setter
    def lengthrange(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def nsample(self) -> int:
        ...
    @nsample.setter
    def nsample(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def userdata(self) -> MjDoubleVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
class MjsBody:
    alt: MjsOrientation
    childclass: str
    classname: MjsDefault
    ialt: MjsOrientation
    info: str
    name: str
    plugin: MjsPlugin
    sleep: mujoco._enums.mjtSleepPolicy
    def add_body(self, default: MjsDefault = None, name: str | None = None, childclass: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mass: typing.SupportsFloat | typing.SupportsIndex | None = None, ipos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, iquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, iaxisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ixyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, izaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ieuler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fullinertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mocap: typing.SupportsInt | typing.SupportsIndex | None = None, gravcomp: typing.SupportsFloat | typing.SupportsIndex | None = None, sleep: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, explicitinertial: typing.SupportsInt | typing.SupportsIndex | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> MjsBody:
        """
              Add body to spec.
        
              Args:
                name: str
                childclass: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                mass: float
                ipos: list[float]
                iquat: list[float]
                inertia: list[float]
                iaxisangle: list[float]
                ixyaxes: list[float]
                izaxis: list[float]
                ieuler: list[float]
                fullinertia: list[float]
                mocap: int
                gravcomp: float
                sleep: int
                userdata: list[float]
                explicitinertial: int
                plugin: MjsPlugin
                info: str
        """
    def add_camera(self, default: MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mode: typing.SupportsInt | typing.SupportsIndex | None = None, targetbody: str | None = None, proj: typing.SupportsInt | typing.SupportsIndex | None = None, resolution: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, output: typing.SupportsInt | typing.SupportsIndex | None = None, fovy: typing.SupportsFloat | typing.SupportsIndex | None = None, ipd: typing.SupportsFloat | typing.SupportsIndex | None = None, intrinsic: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, sensor_size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, focal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, focal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, principal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, principal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsCamera:
        """
              Add camera to spec.
        
              Args:
                name: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                mode: int
                targetbody: str
                proj: int
                resolution: list[float]
                output: int
                fovy: float
                ipd: float
                intrinsic: list[float]
                sensor_size: list[float]
                focal_length: list[float]
                focal_pixel: list[float]
                principal_length: list[float]
                principal_pixel: list[float]
                userdata: list[float]
                info: str
        """
    def add_frame(self, default: MjsFrame = None, name: str | None = None, childclass: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsFrame:
        """
              Add frame to spec.
        
              Args:
                name: str
                childclass: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                info: str
        """
    def add_freejoint(self, **kwargs) -> MjsJoint:
        ...
    def add_geom(self, default: MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: typing.Any | None = None, contype: typing.SupportsInt | typing.SupportsIndex | None = None, conaffinity: typing.SupportsInt | typing.SupportsIndex | None = None, condim: typing.SupportsInt | typing.SupportsIndex | None = None, priority: typing.SupportsInt | typing.SupportsIndex | None = None, friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solmix: typing.SupportsFloat | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, gap: typing.SupportsFloat | typing.SupportsIndex | None = None, mass: typing.SupportsFloat | typing.SupportsIndex | None = None, density: typing.SupportsFloat | typing.SupportsIndex | None = None, typeinertia: typing.SupportsInt | typing.SupportsIndex | None = None, fluid_ellipsoid: typing.SupportsInt | typing.SupportsIndex | None = None, fluid_coefs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, material: str | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, hfieldname: str | None = None, meshname: str | None = None, fitscale: typing.SupportsFloat | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> MjsGeom:
        """
              Add geom to spec.
        
              Args:
                name: str
                type: int
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                fromto: list[float]
                size: Optional[list[float]]
                contype: int
                conaffinity: int
                condim: int
                priority: int
                friction: list[float]
                solmix: float
                solref: list[float]
                solimp: list[float]
                margin: float
                gap: float
                mass: float
                density: float
                typeinertia: int
                fluid_ellipsoid: int
                fluid_coefs: list[float]
                material: str
                rgba: list[float]
                group: int
                hfieldname: str
                meshname: str
                fitscale: float
                userdata: list[float]
                plugin: MjsPlugin
                info: str
        """
    def add_joint(self, default: MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ref: typing.SupportsFloat | typing.SupportsIndex | None = None, align: typing.SupportsInt | typing.SupportsIndex | None = None, stiffness: typing.Any | None = None, springref: typing.SupportsFloat | typing.SupportsIndex | None = None, springdamper: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, limited: typing.SupportsInt | typing.SupportsIndex | None = None, range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = None, actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, armature: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: typing.Any | None = None, frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, actgravcomp: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsJoint:
        """
              Add joint to spec.
        
              Args:
                name: str
                type: int
                pos: list[float]
                axis: list[float]
                ref: float
                align: int
                stiffness: Optional[list[float]]
                springref: float
                springdamper: list[float]
                limited: int
                range: list[float]
                margin: float
                solref_limit: list[float]
                solimp_limit: list[float]
                actfrclimited: int
                actfrcrange: list[float]
                armature: float
                damping: Optional[list[float]]
                frictionloss: float
                solref_friction: list[float]
                solimp_friction: list[float]
                group: int
                actgravcomp: int
                userdata: list[float]
                info: str
        """
    def add_light(self, default: MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, dir: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mode: typing.SupportsInt | typing.SupportsIndex | None = None, targetbody: str | None = None, active: typing.SupportsInt | typing.SupportsIndex | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, texture: str | None = None, castshadow: typing.SupportsInt | typing.SupportsIndex | None = None, bulbradius: typing.SupportsFloat | typing.SupportsIndex | None = None, intensity: typing.SupportsFloat | typing.SupportsIndex | None = None, range: typing.SupportsFloat | typing.SupportsIndex | None = None, attenuation: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, cutoff: typing.SupportsFloat | typing.SupportsIndex | None = None, exponent: typing.SupportsFloat | typing.SupportsIndex | None = None, ambient: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, diffuse: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, specular: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsLight:
        """
              Add light to spec.
        
              Args:
                name: str
                pos: list[float]
                dir: list[float]
                mode: int
                targetbody: str
                active: int
                type: int
                texture: str
                castshadow: int
                bulbradius: float
                intensity: float
                range: float
                attenuation: list[float]
                cutoff: float
                exponent: float
                ambient: list[float]
                diffuse: list[float]
                specular: list[float]
                info: str
        """
    def add_site(self, default: MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: typing.Any | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, material: str | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsSite:
        """
              Add site to spec.
        
              Args:
                name: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                fromto: list[float]
                size: Optional[list[float]]
                type: int
                material: str
                group: int
                rgba: list[float]
                userdata: list[float]
                info: str
        """
    def attach_frame(self, frame: MjsFrame, prefix: str | None = None, suffix: str | None = None) -> MjsFrame:
        ...
    @typing.overload
    def find_all(self, arg0: mujoco._enums.mjtObj) -> list:
        ...
    @typing.overload
    def find_all(self, arg0: str) -> list:
        ...
    def find_child(self, arg0: str) -> MjsBody:
        ...
    def first_body(self) -> MjsBody:
        ...
    def first_camera(self) -> MjsCamera:
        ...
    def first_frame(self) -> MjsFrame:
        ...
    def first_geom(self) -> MjsGeom:
        ...
    def first_joint(self) -> MjsJoint:
        ...
    def first_light(self) -> MjsLight:
        ...
    def first_site(self) -> MjsSite:
        ...
    def next_body(self, arg0: MjsBody) -> MjsBody:
        ...
    def next_camera(self, arg0: MjsCamera) -> MjsCamera:
        ...
    def next_frame(self, arg0: MjsFrame) -> MjsFrame:
        ...
    def next_geom(self, arg0: MjsGeom) -> MjsGeom:
        ...
    def next_joint(self, arg0: MjsJoint) -> MjsJoint:
        ...
    def next_light(self, arg0: MjsLight) -> MjsLight:
        ...
    def next_site(self, arg0: MjsSite) -> MjsSite:
        ...
    def set_frame(self, arg0: MjsFrame) -> None:
        ...
    def to_frame(self) -> MjsFrame:
        ...
    @property
    def bodies(self) -> list:
        ...
    @property
    def cameras(self) -> list:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def explicitinertial(self) -> int:
        ...
    @explicitinertial.setter
    def explicitinertial(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def frame(self) -> MjsFrame:
        ...
    @property
    def frames(self) -> list:
        ...
    @property
    def fullinertia(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"]:
        ...
    @fullinertia.setter
    def fullinertia(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]"]) -> None:
        ...
    @property
    def geoms(self) -> list:
        ...
    @property
    def gravcomp(self) -> float:
        ...
    @gravcomp.setter
    def gravcomp(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def inertia(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @inertia.setter
    def inertia(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def ipos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @ipos.setter
    def ipos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def iquat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @iquat.setter
    def iquat(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def joints(self) -> list:
        ...
    @property
    def lights(self) -> list:
        ...
    @property
    def mass(self) -> float:
        ...
    @mass.setter
    def mass(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mocap(self) -> int:
        ...
    @mocap.setter
    def mocap(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def parent(self) -> MjsBody:
        ...
    @property
    def pos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @pos.setter
    def pos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def quat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @quat.setter
    def quat(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def sites(self) -> list:
        ...
    @property
    def userdata(self) -> MjDoubleVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
class MjsCamera:
    alt: MjsOrientation
    classname: MjsDefault
    info: str
    mode: mujoco._enums.mjtCamLight
    name: str
    proj: mujoco._enums.mjtProjection
    targetbody: str
    def set_frame(self, arg0: MjsFrame) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def focal_length(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]", "flags.writeable"]:
        ...
    @focal_length.setter
    def focal_length(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]"]) -> None:
        ...
    @property
    def focal_pixel(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]", "flags.writeable"]:
        ...
    @focal_pixel.setter
    def focal_pixel(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]"]) -> None:
        ...
    @property
    def fovy(self) -> float:
        ...
    @fovy.setter
    def fovy(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def frame(self) -> MjsFrame:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def intrinsic(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @intrinsic.setter
    def intrinsic(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def ipd(self) -> float:
        ...
    @ipd.setter
    def ipd(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def output(self) -> int:
        ...
    @output.setter
    def output(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def parent(self) -> MjsBody:
        ...
    @property
    def pos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @pos.setter
    def pos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def principal_length(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]", "flags.writeable"]:
        ...
    @principal_length.setter
    def principal_length(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]"]) -> None:
        ...
    @property
    def principal_pixel(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]", "flags.writeable"]:
        ...
    @principal_pixel.setter
    def principal_pixel(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]"]) -> None:
        ...
    @property
    def quat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @quat.setter
    def quat(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def resolution(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[2, 1]", "flags.writeable"]:
        ...
    @resolution.setter
    def resolution(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[2, 1]"]) -> None:
        ...
    @property
    def sensor_size(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]", "flags.writeable"]:
        ...
    @sensor_size.setter
    def sensor_size(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def userdata(self) -> MjDoubleVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
class MjsCompiler:
    LRopt: mujoco._structs.MjLROpt
    meshdir: str
    texturedir: str
    @property
    def alignfree(self) -> int:
        ...
    @alignfree.setter
    def alignfree(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def autolimits(self) -> int:
        ...
    @autolimits.setter
    def autolimits(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def balanceinertia(self) -> int:
        ...
    @balanceinertia.setter
    def balanceinertia(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def boundinertia(self) -> float:
        ...
    @boundinertia.setter
    def boundinertia(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def boundmass(self) -> float:
        ...
    @boundmass.setter
    def boundmass(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def degree(self) -> int:
        ...
    @degree.setter
    def degree(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def discardvisual(self) -> int:
        ...
    @discardvisual.setter
    def discardvisual(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def eulerseq(self) -> MjCharVec:
        ...
    @eulerseq.setter
    def eulerseq(self, arg1: typing.Any) -> None:
        ...
    @property
    def fitaabb(self) -> int:
        ...
    @fitaabb.setter
    def fitaabb(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def fusestatic(self) -> int:
        ...
    @fusestatic.setter
    def fusestatic(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def inertiafromgeom(self) -> int:
        ...
    @inertiafromgeom.setter
    def inertiafromgeom(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def inertiagrouprange(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[2, 1]", "flags.writeable"]:
        ...
    @inertiagrouprange.setter
    def inertiagrouprange(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[2, 1]"]) -> None:
        ...
    @property
    def saveinertial(self) -> int:
        ...
    @saveinertial.setter
    def saveinertial(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def settotalmass(self) -> float:
        ...
    @settotalmass.setter
    def settotalmass(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def usethread(self) -> int:
        ...
    @usethread.setter
    def usethread(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class MjsDefault:
    actuator: MjsActuator
    camera: MjsCamera
    equality: MjsEquality
    flex: MjsFlex
    geom: MjsGeom
    joint: MjsJoint
    light: MjsLight
    material: MjsMaterial
    mesh: MjsMesh
    name: str
    pair: MjsPair
    site: MjsSite
    tendon: MjsTendon
class MjsElement:
    pass
class MjsEquality:
    classname: MjsDefault
    info: str
    name: str
    name1: str
    name2: str
    objtype: mujoco._enums.mjtObj
    type: mujoco._enums.mjtEq
    @property
    def active(self) -> int:
        ...
    @active.setter
    def active(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def data(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[11, 1]", "flags.writeable"]:
        ...
    @data.setter
    def data(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[11, 1]"]) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def solimp(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @solimp.setter
    def solimp(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def solref(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solref.setter
    def solref(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
class MjsExclude:
    bodyname1: str
    bodyname2: str
    info: str
    name: str
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def signature(self) -> int:
        ...
class MjsFlex:
    info: str
    material: str
    name: str
    @property
    def activelayers(self) -> int:
        ...
    @activelayers.setter
    def activelayers(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def conaffinity(self) -> int:
        ...
    @conaffinity.setter
    def conaffinity(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def condim(self) -> int:
        ...
    @condim.setter
    def condim(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def contype(self) -> int:
        ...
    @contype.setter
    def contype(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def damping(self) -> float:
        ...
    @damping.setter
    def damping(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dim(self) -> int:
        ...
    @dim.setter
    def dim(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def edgedamping(self) -> float:
        ...
    @edgedamping.setter
    def edgedamping(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def edgestiffness(self) -> float:
        ...
    @edgestiffness.setter
    def edgestiffness(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def elastic2d(self) -> int:
        ...
    @elastic2d.setter
    def elastic2d(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def elem(self) -> MjIntVec:
        ...
    @elem.setter
    def elem(self, arg1: typing.Any) -> None:
        ...
    @property
    def elemtexcoord(self) -> MjIntVec:
        ...
    @elemtexcoord.setter
    def elemtexcoord(self, arg1: typing.Any) -> None:
        ...
    @property
    def flatskin(self) -> int:
        ...
    @flatskin.setter
    def flatskin(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def friction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @friction.setter
    def friction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def gap(self) -> float:
        ...
    @gap.setter
    def gap(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def group(self) -> int:
        ...
    @group.setter
    def group(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def internal(self) -> int:
        ...
    @internal.setter
    def internal(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def margin(self) -> float:
        ...
    @margin.setter
    def margin(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def node(self) -> MjDoubleVec:
        ...
    @node.setter
    def node(self, arg1: typing.Any) -> None:
        ...
    @property
    def nodebody(self) -> MjStringVec:
        ...
    @nodebody.setter
    def nodebody(self, arg1: typing.Any) -> None:
        ...
    @property
    def passive(self) -> int:
        ...
    @passive.setter
    def passive(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def poisson(self) -> float:
        ...
    @poisson.setter
    def poisson(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def priority(self) -> int:
        ...
    @priority.setter
    def priority(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def radius(self) -> float:
        ...
    @radius.setter
    def radius(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rgba(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @rgba.setter
    def rgba(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def selfcollide(self) -> int:
        ...
    @selfcollide.setter
    def selfcollide(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def size(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @size.setter
    def size(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def solimp(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @solimp.setter
    def solimp(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def solmix(self) -> float:
        ...
    @solmix.setter
    def solmix(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def solref(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solref.setter
    def solref(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def texcoord(self) -> MjFloatVec:
        ...
    @texcoord.setter
    def texcoord(self, arg1: typing.Any) -> None:
        ...
    @property
    def thickness(self) -> float:
        ...
    @thickness.setter
    def thickness(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def vert(self) -> MjDoubleVec:
        ...
    @vert.setter
    def vert(self, arg1: typing.Any) -> None:
        ...
    @property
    def vertbody(self) -> MjStringVec:
        ...
    @vertbody.setter
    def vertbody(self, arg1: typing.Any) -> None:
        ...
    @property
    def young(self) -> float:
        ...
    @young.setter
    def young(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MjsFrame:
    alt: MjsOrientation
    childclass: str
    info: str
    name: str
    def add_body(self, default: MjsDefault = None, name: str | None = None, childclass: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mass: typing.SupportsFloat | typing.SupportsIndex | None = None, ipos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, iquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, iaxisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ixyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, izaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ieuler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fullinertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mocap: typing.SupportsInt | typing.SupportsIndex | None = None, gravcomp: typing.SupportsFloat | typing.SupportsIndex | None = None, sleep: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, explicitinertial: typing.SupportsInt | typing.SupportsIndex | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> MjsBody:
        """
              Add body to spec.
        
              Args:
                name: str
                childclass: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                mass: float
                ipos: list[float]
                iquat: list[float]
                inertia: list[float]
                iaxisangle: list[float]
                ixyaxes: list[float]
                izaxis: list[float]
                ieuler: list[float]
                fullinertia: list[float]
                mocap: int
                gravcomp: float
                sleep: int
                userdata: list[float]
                explicitinertial: int
                plugin: MjsPlugin
                info: str
        """
    def add_camera(self, default: MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mode: typing.SupportsInt | typing.SupportsIndex | None = None, targetbody: str | None = None, proj: typing.SupportsInt | typing.SupportsIndex | None = None, resolution: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, output: typing.SupportsInt | typing.SupportsIndex | None = None, fovy: typing.SupportsFloat | typing.SupportsIndex | None = None, ipd: typing.SupportsFloat | typing.SupportsIndex | None = None, intrinsic: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, sensor_size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, focal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, focal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, principal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, principal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsCamera:
        """
              Add camera to spec.
        
              Args:
                name: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                mode: int
                targetbody: str
                proj: int
                resolution: list[float]
                output: int
                fovy: float
                ipd: float
                intrinsic: list[float]
                sensor_size: list[float]
                focal_length: list[float]
                focal_pixel: list[float]
                principal_length: list[float]
                principal_pixel: list[float]
                userdata: list[float]
                info: str
        """
    def add_frame(self, default: MjsFrame = None, name: str | None = None, childclass: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsFrame:
        """
              Add frame to spec.
        
              Args:
                name: str
                childclass: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                info: str
        """
    def add_geom(self, default: MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: typing.Any | None = None, contype: typing.SupportsInt | typing.SupportsIndex | None = None, conaffinity: typing.SupportsInt | typing.SupportsIndex | None = None, condim: typing.SupportsInt | typing.SupportsIndex | None = None, priority: typing.SupportsInt | typing.SupportsIndex | None = None, friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solmix: typing.SupportsFloat | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, gap: typing.SupportsFloat | typing.SupportsIndex | None = None, mass: typing.SupportsFloat | typing.SupportsIndex | None = None, density: typing.SupportsFloat | typing.SupportsIndex | None = None, typeinertia: typing.SupportsInt | typing.SupportsIndex | None = None, fluid_ellipsoid: typing.SupportsInt | typing.SupportsIndex | None = None, fluid_coefs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, material: str | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, hfieldname: str | None = None, meshname: str | None = None, fitscale: typing.SupportsFloat | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> MjsGeom:
        """
              Add geom to spec.
        
              Args:
                name: str
                type: int
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                fromto: list[float]
                size: Optional[list[float]]
                contype: int
                conaffinity: int
                condim: int
                priority: int
                friction: list[float]
                solmix: float
                solref: list[float]
                solimp: list[float]
                margin: float
                gap: float
                mass: float
                density: float
                typeinertia: int
                fluid_ellipsoid: int
                fluid_coefs: list[float]
                material: str
                rgba: list[float]
                group: int
                hfieldname: str
                meshname: str
                fitscale: float
                userdata: list[float]
                plugin: MjsPlugin
                info: str
        """
    def add_joint(self, default: MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ref: typing.SupportsFloat | typing.SupportsIndex | None = None, align: typing.SupportsInt | typing.SupportsIndex | None = None, stiffness: typing.Any | None = None, springref: typing.SupportsFloat | typing.SupportsIndex | None = None, springdamper: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, limited: typing.SupportsInt | typing.SupportsIndex | None = None, range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = None, actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, armature: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: typing.Any | None = None, frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, actgravcomp: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsJoint:
        """
              Add joint to spec.
        
              Args:
                name: str
                type: int
                pos: list[float]
                axis: list[float]
                ref: float
                align: int
                stiffness: Optional[list[float]]
                springref: float
                springdamper: list[float]
                limited: int
                range: list[float]
                margin: float
                solref_limit: list[float]
                solimp_limit: list[float]
                actfrclimited: int
                actfrcrange: list[float]
                armature: float
                damping: Optional[list[float]]
                frictionloss: float
                solref_friction: list[float]
                solimp_friction: list[float]
                group: int
                actgravcomp: int
                userdata: list[float]
                info: str
        """
    def add_light(self, default: MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, dir: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mode: typing.SupportsInt | typing.SupportsIndex | None = None, targetbody: str | None = None, active: typing.SupportsInt | typing.SupportsIndex | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, texture: str | None = None, castshadow: typing.SupportsInt | typing.SupportsIndex | None = None, bulbradius: typing.SupportsFloat | typing.SupportsIndex | None = None, intensity: typing.SupportsFloat | typing.SupportsIndex | None = None, range: typing.SupportsFloat | typing.SupportsIndex | None = None, attenuation: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, cutoff: typing.SupportsFloat | typing.SupportsIndex | None = None, exponent: typing.SupportsFloat | typing.SupportsIndex | None = None, ambient: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, diffuse: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, specular: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsLight:
        """
              Add light to spec.
        
              Args:
                name: str
                pos: list[float]
                dir: list[float]
                mode: int
                targetbody: str
                active: int
                type: int
                texture: str
                castshadow: int
                bulbradius: float
                intensity: float
                range: float
                attenuation: list[float]
                cutoff: float
                exponent: float
                ambient: list[float]
                diffuse: list[float]
                specular: list[float]
                info: str
        """
    def add_site(self, default: MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: typing.Any | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, material: str | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> MjsSite:
        """
              Add site to spec.
        
              Args:
                name: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                fromto: list[float]
                size: Optional[list[float]]
                type: int
                material: str
                group: int
                rgba: list[float]
                userdata: list[float]
                info: str
        """
    def attach_body(self, body: MjsBody, prefix: str | None = None, suffix: str | None = None) -> MjsBody:
        ...
    def set_frame(self, arg0: MjsFrame) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def frame(self) -> MjsFrame:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def parent(self) -> MjsBody:
        ...
    @property
    def pos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @pos.setter
    def pos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def quat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @quat.setter
    def quat(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
class MjsGeom:
    alt: MjsOrientation
    classname: MjsDefault
    hfieldname: str
    info: str
    material: str
    meshname: str
    name: str
    plugin: MjsPlugin
    type: mujoco._enums.mjtGeom
    typeinertia: mujoco._enums.mjtGeomInertia
    def set_frame(self, arg0: MjsFrame) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def conaffinity(self) -> int:
        ...
    @conaffinity.setter
    def conaffinity(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def condim(self) -> int:
        ...
    @condim.setter
    def condim(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def contype(self) -> int:
        ...
    @contype.setter
    def contype(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def density(self) -> float:
        ...
    @density.setter
    def density(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fitscale(self) -> float:
        ...
    @fitscale.setter
    def fitscale(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fluid_coefs(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @fluid_coefs.setter
    def fluid_coefs(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def fluid_ellipsoid(self) -> float:
        ...
    @fluid_ellipsoid.setter
    def fluid_ellipsoid(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def frame(self) -> MjsFrame:
        ...
    @property
    def friction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @friction.setter
    def friction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def fromto(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"]:
        ...
    @fromto.setter
    def fromto(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]"]) -> None:
        ...
    @property
    def gap(self) -> float:
        ...
    @gap.setter
    def gap(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def group(self) -> int:
        ...
    @group.setter
    def group(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def margin(self) -> float:
        ...
    @margin.setter
    def margin(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mass(self) -> float:
        ...
    @mass.setter
    def mass(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def parent(self) -> MjsBody:
        ...
    @property
    def pos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @pos.setter
    def pos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def priority(self) -> int:
        ...
    @priority.setter
    def priority(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def quat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @quat.setter
    def quat(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def rgba(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @rgba.setter
    def rgba(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def size(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @size.setter
    def size(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def solimp(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @solimp.setter
    def solimp(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def solmix(self) -> float:
        ...
    @solmix.setter
    def solmix(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def solref(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solref.setter
    def solref(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def userdata(self) -> MjDoubleVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
class MjsHField:
    content_type: str
    file: str
    info: str
    name: str
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def ncol(self) -> int:
        ...
    @ncol.setter
    def ncol(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nrow(self) -> int:
        ...
    @nrow.setter
    def nrow(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def size(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @size.setter
    def size(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def userdata(self) -> MjFloatVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
class MjsJoint:
    classname: MjsDefault
    info: str
    name: str
    type: mujoco._enums.mjtJoint
    def set_frame(self, arg0: MjsFrame) -> None:
        ...
    @property
    def actfrclimited(self) -> int:
        ...
    @actfrclimited.setter
    def actfrclimited(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def actfrcrange(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @actfrcrange.setter
    def actfrcrange(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def actgravcomp(self) -> int:
        ...
    @actgravcomp.setter
    def actgravcomp(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def align(self) -> int:
        ...
    @align.setter
    def align(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def armature(self) -> float:
        ...
    @armature.setter
    def armature(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def axis(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @axis.setter
    def axis(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def damping(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @damping.setter
    def damping(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def frame(self) -> MjsFrame:
        ...
    @property
    def frictionloss(self) -> float:
        ...
    @frictionloss.setter
    def frictionloss(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def group(self) -> int:
        ...
    @group.setter
    def group(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def limited(self) -> int:
        ...
    @limited.setter
    def limited(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def margin(self) -> float:
        ...
    @margin.setter
    def margin(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def parent(self) -> MjsBody:
        ...
    @property
    def pos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @pos.setter
    def pos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def range(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @range.setter
    def range(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def ref(self) -> float:
        ...
    @ref.setter
    def ref(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def solimp_friction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @solimp_friction.setter
    def solimp_friction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def solimp_limit(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @solimp_limit.setter
    def solimp_limit(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def solref_friction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solref_friction.setter
    def solref_friction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def solref_limit(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solref_limit.setter
    def solref_limit(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def springdamper(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @springdamper.setter
    def springdamper(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def springref(self) -> float:
        ...
    @springref.setter
    def springref(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def stiffness(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @stiffness.setter
    def stiffness(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def userdata(self) -> MjDoubleVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
class MjsKey:
    info: str
    name: str
    @property
    def act(self) -> MjDoubleVec:
        ...
    @act.setter
    def act(self, arg1: typing.Any) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def ctrl(self) -> MjDoubleVec:
        ...
    @ctrl.setter
    def ctrl(self, arg1: typing.Any) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def mpos(self) -> MjDoubleVec:
        ...
    @mpos.setter
    def mpos(self, arg1: typing.Any) -> None:
        ...
    @property
    def mquat(self) -> MjDoubleVec:
        ...
    @mquat.setter
    def mquat(self, arg1: typing.Any) -> None:
        ...
    @property
    def qpos(self) -> MjDoubleVec:
        ...
    @qpos.setter
    def qpos(self, arg1: typing.Any) -> None:
        ...
    @property
    def qvel(self) -> MjDoubleVec:
        ...
    @qvel.setter
    def qvel(self, arg1: typing.Any) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def time(self) -> float:
        ...
    @time.setter
    def time(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MjsLight:
    classname: MjsDefault
    info: str
    mode: mujoco._enums.mjtCamLight
    name: str
    targetbody: str
    texture: str
    type: mujoco._enums.mjtLightType
    def set_frame(self, arg0: MjsFrame) -> None:
        ...
    @property
    def active(self) -> int:
        ...
    @active.setter
    def active(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ambient(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]", "flags.writeable"]:
        ...
    @ambient.setter
    def ambient(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]) -> None:
        ...
    @property
    def attenuation(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]", "flags.writeable"]:
        ...
    @attenuation.setter
    def attenuation(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]) -> None:
        ...
    @property
    def bulbradius(self) -> float:
        ...
    @bulbradius.setter
    def bulbradius(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def castshadow(self) -> int:
        ...
    @castshadow.setter
    def castshadow(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def cutoff(self) -> float:
        ...
    @cutoff.setter
    def cutoff(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def diffuse(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]", "flags.writeable"]:
        ...
    @diffuse.setter
    def diffuse(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]) -> None:
        ...
    @property
    def dir(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @dir.setter
    def dir(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def exponent(self) -> float:
        ...
    @exponent.setter
    def exponent(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def frame(self) -> MjsFrame:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def intensity(self) -> float:
        ...
    @intensity.setter
    def intensity(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def parent(self) -> MjsBody:
        ...
    @property
    def pos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @pos.setter
    def pos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def range(self) -> float:
        ...
    @range.setter
    def range(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def specular(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]", "flags.writeable"]:
        ...
    @specular.setter
    def specular(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]) -> None:
        ...
class MjsMaterial:
    classname: MjsDefault
    info: str
    name: str
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def emission(self) -> float:
        ...
    @emission.setter
    def emission(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def metallic(self) -> float:
        ...
    @metallic.setter
    def metallic(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def reflectance(self) -> float:
        ...
    @reflectance.setter
    def reflectance(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rgba(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @rgba.setter
    def rgba(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def roughness(self) -> float:
        ...
    @roughness.setter
    def roughness(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def shininess(self) -> float:
        ...
    @shininess.setter
    def shininess(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def specular(self) -> float:
        ...
    @specular.setter
    def specular(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def texrepeat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]", "flags.writeable"]:
        ...
    @texrepeat.setter
    def texrepeat(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[2, 1]"]) -> None:
        ...
    @property
    def textures(self) -> MjStringVec:
        ...
    @textures.setter
    def textures(self, arg1: typing.Any) -> None:
        ...
    @property
    def texuniform(self) -> int:
        ...
    @texuniform.setter
    def texuniform(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class MjsMesh:
    classname: MjsDefault
    content_type: str
    file: str
    inertia: mujoco._enums.mjtMeshInertia
    info: str
    material: str
    name: str
    plugin: MjsPlugin
    def make_cone(self, nedge: typing.SupportsInt | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def make_hemisphere(self, resolution: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def make_plate(self, resolution: typing.Annotated[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], "FixedSize(2)"] = [0, 0]) -> None:
        ...
    def make_sphere(self, subdivision: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def make_supersphere(self, resolution: typing.SupportsInt | typing.SupportsIndex, e: typing.SupportsFloat | typing.SupportsIndex, n: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def make_supertorus(self, resolution: typing.SupportsInt | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, s: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def make_wedge(self, resolution: typing.Annotated[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], "FixedSize(2)"] = [0, 0], fov: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"] = [0.0, 0.0], gamma: typing.SupportsFloat | typing.SupportsIndex = 0) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def maxhullvert(self) -> int:
        ...
    @maxhullvert.setter
    def maxhullvert(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def needsdf(self) -> int:
        ...
    @needsdf.setter
    def needsdf(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def refpos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @refpos.setter
    def refpos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def refquat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @refquat.setter
    def refquat(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def scale(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @scale.setter
    def scale(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def smoothnormal(self) -> int:
        ...
    @smoothnormal.setter
    def smoothnormal(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def userface(self) -> MjIntVec:
        ...
    @userface.setter
    def userface(self, arg1: typing.Any) -> None:
        ...
    @property
    def userfacenormal(self) -> MjIntVec:
        ...
    @userfacenormal.setter
    def userfacenormal(self, arg1: typing.Any) -> None:
        ...
    @property
    def userfacetexcoord(self) -> MjIntVec:
        ...
    @userfacetexcoord.setter
    def userfacetexcoord(self, arg1: typing.Any) -> None:
        ...
    @property
    def usernormal(self) -> MjFloatVec:
        ...
    @usernormal.setter
    def usernormal(self, arg1: typing.Any) -> None:
        ...
    @property
    def usertexcoord(self) -> MjFloatVec:
        ...
    @usertexcoord.setter
    def usertexcoord(self, arg1: typing.Any) -> None:
        ...
    @property
    def uservert(self) -> MjFloatVec:
        ...
    @uservert.setter
    def uservert(self, arg1: typing.Any) -> None:
        ...
class MjsNumeric:
    info: str
    name: str
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def data(self) -> MjDoubleVec:
        ...
    @data.setter
    def data(self, arg1: typing.Any) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
    @size.setter
    def size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class MjsOrientation:
    type: mujoco._enums.mjtOrientation
    @property
    def axisangle(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @axisangle.setter
    def axisangle(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def euler(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @euler.setter
    def euler(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def xyaxes(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"]:
        ...
    @xyaxes.setter
    def xyaxes(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]"]) -> None:
        ...
    @property
    def zaxis(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @zaxis.setter
    def zaxis(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
class MjsPair:
    classname: MjsDefault
    geomname1: str
    geomname2: str
    info: str
    name: str
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def condim(self) -> int:
        ...
    @condim.setter
    def condim(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def friction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @friction.setter
    def friction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def gap(self) -> float:
        ...
    @gap.setter
    def gap(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def margin(self) -> float:
        ...
    @margin.setter
    def margin(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def solimp(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @solimp.setter
    def solimp(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def solref(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solref.setter
    def solref(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def solreffriction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solreffriction.setter
    def solreffriction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
class MjsPlugin:
    config: dict
    info: str
    name: str
    plugin_name: str
    @property
    def active(self) -> int:
        ...
    @active.setter
    def active(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def id(self) -> int:
        ...
    @id.setter
    def id(self, arg1: MjsPlugin) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
class MjsSensor:
    datatype: mujoco._enums.mjtDataType
    info: str
    name: str
    needstage: mujoco._enums.mjtStage
    objname: str
    objtype: mujoco._enums.mjtObj
    plugin: MjsPlugin
    refname: str
    reftype: mujoco._enums.mjtObj
    type: mujoco._enums.mjtSensor
    def get_data_size(self) -> int:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def cutoff(self) -> float:
        ...
    @cutoff.setter
    def cutoff(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def delay(self) -> float:
        ...
    @delay.setter
    def delay(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dim(self) -> int:
        ...
    @dim.setter
    def dim(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def interp(self) -> int:
        ...
    @interp.setter
    def interp(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def interval(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @interval.setter
    def interval(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def intprm(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[3, 1]", "flags.writeable"]:
        ...
    @intprm.setter
    def intprm(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[3, 1]"]) -> None:
        ...
    @property
    def noise(self) -> float:
        ...
    @noise.setter
    def noise(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def nsample(self) -> int:
        ...
    @nsample.setter
    def nsample(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def userdata(self) -> MjDoubleVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
class MjsSite:
    alt: MjsOrientation
    classname: MjsDefault
    info: str
    material: str
    name: str
    type: mujoco._enums.mjtGeom
    def attach_body(self, body: MjsBody, prefix: str | None = None, suffix: str | None = None) -> MjsBody:
        ...
    def set_frame(self, arg0: MjsFrame) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def frame(self) -> MjsFrame:
        ...
    @property
    def fromto(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"]:
        ...
    @fromto.setter
    def fromto(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]"]) -> None:
        ...
    @property
    def group(self) -> int:
        ...
    @group.setter
    def group(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def parent(self) -> MjsBody:
        ...
    @property
    def pos(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @pos.setter
    def pos(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def quat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"]:
        ...
    @quat.setter
    def quat(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"]) -> None:
        ...
    @property
    def rgba(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @rgba.setter
    def rgba(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def size(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @size.setter
    def size(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def userdata(self) -> MjDoubleVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
class MjsSkin:
    file: str
    info: str
    material: str
    name: str
    @property
    def bindpos(self) -> MjFloatVec:
        ...
    @bindpos.setter
    def bindpos(self, arg1: typing.Any) -> None:
        ...
    @property
    def bindquat(self) -> MjFloatVec:
        ...
    @bindquat.setter
    def bindquat(self, arg1: typing.Any) -> None:
        ...
    @property
    def bodyname(self) -> MjStringVec:
        ...
    @bodyname.setter
    def bodyname(self, arg1: typing.Any) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def face(self) -> MjIntVec:
        ...
    @face.setter
    def face(self, arg1: typing.Any) -> None:
        ...
    @property
    def group(self) -> int:
        ...
    @group.setter
    def group(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def inflate(self) -> float:
        ...
    @inflate.setter
    def inflate(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rgba(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @rgba.setter
    def rgba(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def texcoord(self) -> MjFloatVec:
        ...
    @texcoord.setter
    def texcoord(self, arg1: typing.Any) -> None:
        ...
    @property
    def vert(self) -> MjFloatVec:
        ...
    @vert.setter
    def vert(self, arg1: typing.Any) -> None:
        ...
    @property
    def vertid(self) -> list:
        ...
    @vertid.setter
    def vertid(self, arg1: typing.Any) -> None:
        ...
    @property
    def vertweight(self) -> list:
        ...
    @vertweight.setter
    def vertweight(self, arg1: typing.Any) -> None:
        ...
class MjsTendon:
    info: str
    material: str
    name: str
    def default(self) -> MjsDefault:
        ...
    def wrap_geom(self, arg0: str, arg1: str) -> MjsWrap:
        ...
    def wrap_joint(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> MjsWrap:
        ...
    def wrap_pulley(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> MjsWrap:
        ...
    def wrap_site(self, arg0: str) -> MjsWrap:
        ...
    @property
    def actfrclimited(self) -> int:
        ...
    @actfrclimited.setter
    def actfrclimited(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def actfrcrange(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @actfrcrange.setter
    def actfrcrange(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def armature(self) -> float:
        ...
    @armature.setter
    def armature(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def damping(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @damping.setter
    def damping(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def frictionloss(self) -> float:
        ...
    @frictionloss.setter
    def frictionloss(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def group(self) -> int:
        ...
    @group.setter
    def group(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def limited(self) -> int:
        ...
    @limited.setter
    def limited(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def margin(self) -> float:
        ...
    @margin.setter
    def margin(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def path(self) -> MjsTendonPath:
        ...
    @property
    def range(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @range.setter
    def range(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def rgba(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]", "flags.writeable"]:
        ...
    @rgba.setter
    def rgba(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def solimp_friction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @solimp_friction.setter
    def solimp_friction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def solimp_limit(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]", "flags.writeable"]:
        ...
    @solimp_limit.setter
    def solimp_limit(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[5, 1]"]) -> None:
        ...
    @property
    def solref_friction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solref_friction.setter
    def solref_friction(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def solref_limit(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @solref_limit.setter
    def solref_limit(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def springlength(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]", "flags.writeable"]:
        ...
    @springlength.setter
    def springlength(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"]) -> None:
        ...
    @property
    def stiffness(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @stiffness.setter
    def stiffness(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def userdata(self) -> MjDoubleVec:
        ...
    @userdata.setter
    def userdata(self, arg1: typing.Any) -> None:
        ...
    @property
    def width(self) -> float:
        ...
    @width.setter
    def width(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MjsTendonPath:
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> MjsWrap:
        ...
    def __len__(self) -> int:
        ...
class MjsText:
    data: str
    info: str
    name: str
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def signature(self) -> int:
        ...
class MjsTexture:
    colorspace: mujoco._enums.mjtColorSpace
    content_type: str
    data: bytes
    file: str
    info: str
    name: str
    type: mujoco._enums.mjtTexture
    @property
    def builtin(self) -> int:
        ...
    @builtin.setter
    def builtin(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def cubefiles(self) -> MjStringVec:
        ...
    @cubefiles.setter
    def cubefiles(self, arg1: typing.Any) -> None:
        ...
    @property
    def gridlayout(self) -> MjCharVec:
        ...
    @gridlayout.setter
    def gridlayout(self, arg1: typing.Any) -> None:
        ...
    @property
    def gridsize(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[2, 1]", "flags.writeable"]:
        ...
    @gridsize.setter
    def gridsize(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[2, 1]"]) -> None:
        ...
    @property
    def height(self) -> int:
        ...
    @height.setter
    def height(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def hflip(self) -> int:
        ...
    @hflip.setter
    def hflip(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def mark(self) -> int:
        ...
    @mark.setter
    def mark(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def markrgb(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @markrgb.setter
    def markrgb(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def nchannel(self) -> int:
        ...
    @nchannel.setter
    def nchannel(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def random(self) -> float:
        ...
    @random.setter
    def random(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rgb1(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @rgb1.setter
    def rgb1(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def rgb2(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"]:
        ...
    @rgb2.setter
    def rgb2(self, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
    @property
    def vflip(self) -> int:
        ...
    @vflip.setter
    def vflip(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def width(self) -> int:
        ...
    @width.setter
    def width(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class MjsTuple:
    info: str
    name: str
    @property
    def compiler(self) -> MjsCompiler:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def objname(self) -> MjStringVec:
        ...
    @objname.setter
    def objname(self, arg1: typing.Any) -> None:
        ...
    @property
    def objprm(self) -> MjDoubleVec:
        ...
    @objprm.setter
    def objprm(self, arg1: typing.Any) -> None:
        ...
    @property
    def objtype(self) -> MjIntVec:
        ...
    @objtype.setter
    def objtype(self, arg1: typing.Any) -> None:
        ...
    @property
    def signature(self) -> int:
        ...
class MjsWrap:
    info: str
    type: mujoco._enums.mjtWrap
    @property
    def coef(self) -> typing.Any:
        ...
    @property
    def divisor(self) -> typing.Any:
        ...
    @property
    def sidesite(self) -> MjsSite:
        ...
    @property
    def target(self) -> typing.Any:
        ...
