# utils for adapting the imitation library to our use case
# imitation library: https://github.com/HumanCompatibleAI/imitation
# generator library: stable-baselines3

import numpy as np
import os
import functools
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from stable_baselines3.common import monitor, policies
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from envs.pickplace import *
from imitation.util.util import make_seeds


def make_vec_env_robosuite(
    env_name: str,
    obs_keys: Iterable[str],
    *,
    rng: np.random.Generator,
    n_envs: int = 8,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
    sequential_wrapper = None
) -> VecEnv:
    """
    Create a VecEnv for a Robosuite environment.
    """
    # Resolve the spec outside of the subprocess first, so that it is available to
    # subprocesses running `make_env` via automatic pickling.
    # Just to ensure packages are imported and spec is properly resolved
    env = suite.make(env_name, **env_make_kwargs)
    tmp_env = GymWrapper(env, keys=obs_keys)
    tmp_env.close()
    spec = tmp_env.spec
    env_make_kwargs = env_make_kwargs or {}

    def make_env(i: int, this_seed: int) -> gym.Env:
        # Previously, we directly called `gym.make(env_name)`, but running
        # `imitation.scripts.train_adversarial` within `imitation.scripts.parallel`
        # created a weird interaction between Gym and Ray -- `gym.make` would fail
        # inside this function for any of our custom environment unless those
        # environments were also `gym.register()`ed inside `make_env`. Even
        # registering the custom environment in the scope of `make_vec_env` didn't
        # work. For more discussion and hypotheses on this issue see PR #160:
        # https://github.com/HumanCompatibleAI/imitation/pull/160.
        assert env_make_kwargs is not None  # Note: to satisfy mypy
        # assert spec is not None  # Note: to satisfy mypy
        env = suite.make(env_name, **env_make_kwargs)
        env = GymWrapper(env, keys=obs_keys)

        # Seed each environment with a different, non-sequential seed for diversity
        # (even if caller is passing us sequentially-assigned base seeds). int() is
        # necessary to work around gym bug where it chokes on numpy int64s.
        env.reset(seed=int(this_seed))
        # NOTE: we do it here rather than on the final VecEnv, because
        # that would set the same seed for all the environments.

        # Use Monitor to record statistics needed for Baselines algorithms logging
        # Optionally, save to disk
        log_path = None
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, "monitor")
            os.makedirs(log_subdir, exist_ok=True)
            log_path = os.path.join(log_subdir, f"mon{i:03d}")

        env = monitor.Monitor(env, log_path)

        if post_wrappers:
            for wrapper in post_wrappers:
                env = wrapper(env, i)

        return env

    env_seeds = make_seeds(rng, n_envs)
    env_fns: List[Callable[[], gym.Env]] = [
        functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
    ]
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        return SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        return DummyVecEnv(env_fns)
