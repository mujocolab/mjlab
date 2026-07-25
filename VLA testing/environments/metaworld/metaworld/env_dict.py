"""Dictionaries mapping environment name strings to environment classes,
and organising them into various collections and splits for the benchmarks."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Literal
from typing import OrderedDict as Typing_OrderedDict
from typing import Sequence, Union

import numpy as np
from typing_extensions import TypeAlias

from metaworld import envs
from metaworld.sawyer_xyz_env import SawyerXYZEnv

# Utils

EnvDict: TypeAlias = "Typing_OrderedDict[str, type[SawyerXYZEnv]]"
TrainTestEnvDict: TypeAlias = "Typing_OrderedDict[Literal['train', 'test'], EnvDict]"
EnvArgsKwargsDict: TypeAlias = (
    "Dict[str, Dict[Literal['args', 'kwargs'], Union[List, Dict]]]"
)

ENV_CLS_MAP = {
    "assembly-v3": envs.SawyerNutAssemblyEnvV3,
    "basketball-v3": envs.SawyerBasketballEnvV3,
    "bin-picking-v3": envs.SawyerBinPickingEnvV3,
    "box-close-v3": envs.SawyerBoxCloseEnvV3,
    "button-press-topdown-v3": envs.SawyerButtonPressTopdownEnvV3,
    "button-press-topdown-wall-v3": envs.SawyerButtonPressTopdownWallEnvV3,
    "button-press-v3": envs.SawyerButtonPressEnvV3,
    "button-press-wall-v3": envs.SawyerButtonPressWallEnvV3,
    "coffee-button-v3": envs.SawyerCoffeeButtonEnvV3,
    "coffee-pull-v3": envs.SawyerCoffeePullEnvV3,
    "coffee-push-v3": envs.SawyerCoffeePushEnvV3,
    "dial-turn-v3": envs.SawyerDialTurnEnvV3,
    "disassemble-v3": envs.SawyerNutDisassembleEnvV3,
    "door-close-v3": envs.SawyerDoorCloseEnvV3,
    "door-lock-v3": envs.SawyerDoorLockEnvV3,
    "door-open-v3": envs.SawyerDoorEnvV3,
    "door-unlock-v3": envs.SawyerDoorUnlockEnvV3,
    "hand-insert-v3": envs.SawyerHandInsertEnvV3,
    "drawer-close-v3": envs.SawyerDrawerCloseEnvV3,
    "drawer-open-v3": envs.SawyerDrawerOpenEnvV3,
    "faucet-open-v3": envs.SawyerFaucetOpenEnvV3,
    "faucet-close-v3": envs.SawyerFaucetCloseEnvV3,
    "hammer-v3": envs.SawyerHammerEnvV3,
    "handle-press-side-v3": envs.SawyerHandlePressSideEnvV3,
    "handle-press-v3": envs.SawyerHandlePressEnvV3,
    "handle-pull-side-v3": envs.SawyerHandlePullSideEnvV3,
    "handle-pull-v3": envs.SawyerHandlePullEnvV3,
    "lever-pull-v3": envs.SawyerLeverPullEnvV3,
    "peg-insert-side-v3": envs.SawyerPegInsertionSideEnvV3,
    "pick-place-wall-v3": envs.SawyerPickPlaceWallEnvV3,
    "pick-out-of-hole-v3": envs.SawyerPickOutOfHoleEnvV3,
    "reach-v3": envs.SawyerReachEnvV3,
    "push-back-v3": envs.SawyerPushBackEnvV3,
    "push-v3": envs.SawyerPushEnvV3,
    "pick-place-v3": envs.SawyerPickPlaceEnvV3,
    "plate-slide-v3": envs.SawyerPlateSlideEnvV3,
    "plate-slide-side-v3": envs.SawyerPlateSlideSideEnvV3,
    "plate-slide-back-v3": envs.SawyerPlateSlideBackEnvV3,
    "plate-slide-back-side-v3": envs.SawyerPlateSlideBackSideEnvV3,
    "peg-unplug-side-v3": envs.SawyerPegUnplugSideEnvV3,
    "soccer-v3": envs.SawyerSoccerEnvV3,
    "stick-push-v3": envs.SawyerStickPushEnvV3,
    "stick-pull-v3": envs.SawyerStickPullEnvV3,
    "push-wall-v3": envs.SawyerPushWallEnvV3,
    "reach-wall-v3": envs.SawyerReachWallEnvV3,
    "shelf-place-v3": envs.SawyerShelfPlaceEnvV3,
    "sweep-into-v3": envs.SawyerSweepIntoGoalEnvV3,
    "sweep-v3": envs.SawyerSweepEnvV3,
    "window-open-v3": envs.SawyerWindowOpenEnvV3,
    "window-close-v3": envs.SawyerWindowCloseEnvV3,
}


def _get_env_dict(env_names: Sequence[str]) -> EnvDict:
    """Returns an `OrderedDict` containing `(env_name, env_cls)` tuples for the given env_names.

    Args:
        env_names: The environment names

    Returns:
        The appropriate `OrderedDict.
    """
    return OrderedDict([(env_name, ENV_CLS_MAP[env_name]) for env_name in env_names])


def _get_train_test_env_dict(
    train_env_names: Sequence[str], test_env_names: Sequence[str]
) -> TrainTestEnvDict:
    """Returns an `OrderedDict` containing two sub-keys ("train" and "test" at positions 0 and 1),
    each containing the appropriate `OrderedDict` for the train and test classes of the benchmark.

    Args:
        train_env_names: The train environment names.
        test_env_names: The test environment names

    Returns:
        The appropriate `OrderedDict`.
    """
    return OrderedDict(
        (
            ("train", _get_env_dict(train_env_names)),
            ("test", _get_env_dict(test_env_names)),
        )
    )


def _get_args_kwargs(all_envs: EnvDict, env_subset: EnvDict) -> EnvArgsKwargsDict:
    """Returns containing a `dict` of "args" and "kwargs" for each environment in a given list of environments.
    Specifically, sets an empty "args" array and a "kwargs" dictionary with a "task_id" key for each env.

    Args:
        all_envs: The full list of envs
        env_subset: The subset of envs to get args and kwargs for

    Returns:
        The args and kwargs dictionary.
    """
    return {
        key: dict(args=[], kwargs={"task_id": list(all_envs.keys()).index(key)})
        for key, _ in env_subset.items()
    }


def _create_hidden_goal_envs(all_envs: EnvDict) -> EnvDict:
    """Create versions of the environments with the goal hidden.

    Args:
        all_envs: The full list of envs in the benchmark.

    Returns:
        An `EnvDict` where the classes have been modified to hide the goal.
    """
    hidden_goal_envs = {}
    for env_name, env_cls in all_envs.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = True
            del env.sawyer_observation_space
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed=seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        hg_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        hg_env_name = hg_env_name.replace("-", "")
        hg_env_key = f"{env_name}-goal-hidden"
        hg_env_name = f"{hg_env_name}GoalHidden"
        HiddenGoalEnvCls = type(hg_env_name, (env_cls,), d)
        hidden_goal_envs[hg_env_key] = HiddenGoalEnvCls

    return OrderedDict(hidden_goal_envs)


def _create_observable_goal_envs(all_envs: EnvDict) -> EnvDict:
    """Create versions of the environments with the goal observable.

    Args:
        all_envs: The full list of envs in the benchmark.

    Returns:
        An `EnvDict` where the classes have been modified to make the goal observable.
    """
    observable_goal_envs = {}
    for env_name, env_cls in all_envs.items():
        d = {}

        def initialize(env, seed=None, render_mode=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()

            env._partially_observable = False
            env._freeze_rand_vec = False
            del env.sawyer_observation_space
            env._set_task_called = True
            env.render_mode = render_mode
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        og_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        og_env_name = og_env_name.replace("-", "")

        og_env_key = f"{env_name}-goal-observable"
        og_env_name = f"{og_env_name}GoalObservable"
        ObservableGoalEnvCls = type(og_env_name, (env_cls,), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


# V3 DICTS

ALL_V3_ENVIRONMENTS = _get_env_dict(
    [
        "assembly-v3",
        "basketball-v3",
        "bin-picking-v3",
        "box-close-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-v3",
        "button-press-wall-v3",
        "coffee-button-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "dial-turn-v3",
        "disassemble-v3",
        "door-close-v3",
        "door-lock-v3",
        "door-open-v3",
        "door-unlock-v3",
        "hand-insert-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-open-v3",
        "faucet-close-v3",
        "hammer-v3",
        "handle-press-side-v3",
        "handle-press-v3",
        "handle-pull-side-v3",
        "handle-pull-v3",
        "lever-pull-v3",
        "pick-place-wall-v3",
        "pick-out-of-hole-v3",
        "pick-place-v3",
        "plate-slide-v3",
        "plate-slide-side-v3",
        "plate-slide-back-v3",
        "plate-slide-back-side-v3",
        "peg-insert-side-v3",
        "peg-unplug-side-v3",
        "soccer-v3",
        "stick-push-v3",
        "stick-pull-v3",
        "push-v3",
        "push-wall-v3",
        "push-back-v3",
        "reach-v3",
        "reach-wall-v3",
        "shelf-place-v3",
        "sweep-into-v3",
        "sweep-v3",
        "window-open-v3",
        "window-close-v3",
    ]
)


ALL_V3_ENVIRONMENTS_GOAL_HIDDEN = _create_hidden_goal_envs(ALL_V3_ENVIRONMENTS)
ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE = _create_observable_goal_envs(ALL_V3_ENVIRONMENTS)

# MT Dicts

MT10_V3 = _get_env_dict(
    [
        "reach-v3",
        "push-v3",
        "pick-place-v3",
        "door-open-v3",
        "drawer-open-v3",
        "drawer-close-v3",
        "button-press-topdown-v3",
        "peg-insert-side-v3",
        "window-open-v3",
        "window-close-v3",
    ]
)
MT10_V3_ARGS_KWARGS = _get_args_kwargs(ALL_V3_ENVIRONMENTS, MT10_V3)

MT25_V3 = _get_env_dict(
    [
        "reach-v3",
        "push-v3",
        "pick-place-v3",
        "door-open-v3",
        "drawer-open-v3",
        "drawer-close-v3",
        "button-press-topdown-v3",
        "peg-insert-side-v3",
        "window-open-v3",
        "window-close-v3",
        "coffee-pull-v3",
        "pick-out-of-hole-v3",
        "disassemble-v3",
        "pick-place-wall-v3",
        "basketball-v3",
        "stick-pull-v3",
        "button-press-wall-v3",
        "faucet-open-v3",
        "door-lock-v3",
        "lever-pull-v3",
        "sweep-into-v3",
        "faucet-close-v3",
        "coffee-button-v3",
        "button-press-topdown-wall-v3",
        "dial-turn-v3",
    ]
)
MT25_V3_ARGS_KWARGS = _get_args_kwargs(ALL_V3_ENVIRONMENTS, MT25_V3)


MT50_V3 = ALL_V3_ENVIRONMENTS
MT50_V3_ARGS_KWARGS = _get_args_kwargs(ALL_V3_ENVIRONMENTS, MT50_V3)

# ML Dicts

ML1_V3 = _get_train_test_env_dict(
    list(ALL_V3_ENVIRONMENTS.keys()), list(ALL_V3_ENVIRONMENTS.keys())
)
ML1_ARGS_KWARGS = _get_args_kwargs(ALL_V3_ENVIRONMENTS, ML1_V3["train"])

ML10_V3 = _get_train_test_env_dict(
    train_env_names=[
        "reach-v3",
        "push-v3",
        "pick-place-v3",
        "door-open-v3",
        "drawer-close-v3",
        "button-press-topdown-v3",
        "peg-insert-side-v3",
        "window-open-v3",
        "sweep-v3",
        "basketball-v3",
    ],
    test_env_names=[
        "drawer-open-v3",
        "door-close-v3",
        "shelf-place-v3",
        "sweep-into-v3",
        "lever-pull-v3",
    ],
)
ML10_ARGS_KWARGS = {
    "train": _get_args_kwargs(ALL_V3_ENVIRONMENTS, ML10_V3["train"]),
    "test": _get_args_kwargs(ALL_V3_ENVIRONMENTS, ML10_V3["test"]),
}


ML25_V3 = _get_train_test_env_dict(
    train_env_names=[
        "reach-v3",
        "push-v3",
        "pick-place-v3",
        "door-open-v3",
        "drawer-open-v3",
        "drawer-close-v3",
        "button-press-topdown-v3",
        "peg-insert-side-v3",
        "window-open-v3",
        "window-close-v3",
        "coffee-pull-v3",
        "pick-out-of-hole-v3",
        "disassemble-v3",
        "pick-place-wall-v3",
        "basketball-v3",
        "stick-pull-v3",
        "button-press-wall-v3",
        "faucet-open-v3",
        "door-lock-v3",
        "lever-pull-v3",
        "sweep-into-v3",
        "faucet-close-v3",
        "coffee-button-v3",
        "button-press-topdown-wall-v3",
        "dial-turn-v3",
    ],
    test_env_names=[
        "basketball-v3",
        "door-close-v3",
        "shelf-place-v3",
        "sweep-v3",
        "button-press-v3",
    ],
)

ML25_ARGS_KWARGS = {
    "train": _get_args_kwargs(ALL_V3_ENVIRONMENTS, ML25_V3["train"]),
    "test": _get_args_kwargs(ALL_V3_ENVIRONMENTS, ML25_V3["test"]),
}


ML45_V3 = _get_train_test_env_dict(
    train_env_names=[
        "assembly-v3",
        "basketball-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-v3",
        "button-press-wall-v3",
        "coffee-button-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "dial-turn-v3",
        "disassemble-v3",
        "door-close-v3",
        "door-open-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-open-v3",
        "faucet-close-v3",
        "hammer-v3",
        "handle-press-side-v3",
        "handle-press-v3",
        "handle-pull-side-v3",
        "handle-pull-v3",
        "lever-pull-v3",
        "pick-place-wall-v3",
        "pick-out-of-hole-v3",
        "push-back-v3",
        "pick-place-v3",
        "plate-slide-v3",
        "plate-slide-side-v3",
        "plate-slide-back-v3",
        "plate-slide-back-side-v3",
        "peg-insert-side-v3",
        "peg-unplug-side-v3",
        "soccer-v3",
        "stick-push-v3",
        "stick-pull-v3",
        "push-wall-v3",
        "push-v3",
        "reach-wall-v3",
        "reach-v3",
        "shelf-place-v3",
        "sweep-into-v3",
        "sweep-v3",
        "window-open-v3",
        "window-close-v3",
    ],
    test_env_names=[
        "bin-picking-v3",
        "box-close-v3",
        "hand-insert-v3",
        "door-lock-v3",
        "door-unlock-v3",
    ],
)
ML45_ARGS_KWARGS = {
    "train": _get_args_kwargs(ALL_V3_ENVIRONMENTS, ML45_V3["train"]),
    "test": _get_args_kwargs(ALL_V3_ENVIRONMENTS, ML45_V3["test"]),
}
