import random
from functools import partial
from typing import Iterable

import gym
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import lbforaging
import bpush
import rware
import pettingzoo as pz

from mate.utils import wrappers as mwrappers


def _make_parallel_envs(name, parallel_envs, dummy_vecenv, wrappers, seed, max_ep_length, arguments):
    def _env_thunk(env_seed):
        generalisation_envs = False
        for v in arguments.values():
            if not isinstance(v, str) and isinstance(v, Iterable):
                generalisation_envs = True

        if not isinstance(name, str) and isinstance(name, Iterable):
            generalisation_envs = True

        if generalisation_envs:
            if not "GeneralisationWrapper" in wrappers:
                wrappers.insert(0, "GeneralisationWrapper")
            # random starting environment
            args = {}
            for k, v in arguments.items():
                if not isinstance(v, str) and isinstance(v, Iterable):
                    args[k] = random.choice(v)
                else:
                    args[k] = v

            if not isinstance(name, str) and isinstance(name, Iterable):
                init_name = random.choice(name)
            else:
                init_name = name

            env = gym.make(init_name, **args)
        else:
            env = gym.make(name, **arguments)

        if max_ep_length is not None and max_ep_length > 0:
            if not "TimeLimit" in wrappers:
                if "GeneralisationWrapper" in wrappers:
                    idx = wrappers.index("GeneralisationWrapper")
                    wrappers.insert(idx+1, "TimeLimit")
                else:
                    wrappers.insert(0, "TimeLimit")

        for wrapper in wrappers:
            wrap = getattr(mwrappers, wrapper) 
            if wrapper == "GeneralisationWrapper":
                assert generalisation_envs
                env = wrap(env, name, **arguments)
            elif wrapper == "TimeLimit":
                assert max_ep_length is not None
                env = wrap(env, max_ep_length)
            else:
                env = wrap(env)
        env.seed(env_seed)
        return env

    if seed is None:
        seed = random.randint(0, 99999)

    env_thunks = [partial(_env_thunk, seed + i) for i in range(parallel_envs)]
    if dummy_vecenv:
        envs = DummyVecEnv(env_thunks)
        envs.buf_rews = np.zeros(
            (parallel_envs, len(envs.observation_space)), dtype=np.float32
        )
    else:
        envs = SubprocVecEnv(env_thunks, start_method="fork")
    return envs


def _make_env(name, wrappers, seed, max_ep_length, arguments={}):
    generalisation_envs = False
    for v in arguments.values():
        if not isinstance(v, str) and isinstance(v, Iterable):
            generalisation_envs = True

    if not isinstance(name, str) and isinstance(name, Iterable):
        generalisation_envs = True

    if generalisation_envs:
        if not "GeneralisationWrapper" in wrappers:
            wrappers.insert(0, "GeneralisationWrapper")
        # random starting environment
        args = {}
        for k, v in arguments.items():
            if not isinstance(v, str) and isinstance(v, Iterable):
                args[k] = random.choice(v)
            else:
                args[k] = v
        if not isinstance(name, str) and isinstance(name, Iterable):
            init_name = random.choice(name)
        else:
            init_name = name
        env = gym.make(init_name, **args)
    else:
        env = gym.make(name, **arguments)

    if max_ep_length is not None and max_ep_length > 0:
        if not "TimeLimit" in wrappers:
            if "GeneralisationWrapper" in wrappers:
                idx = wrappers.index("GeneralisationWrapper")
                wrappers.insert(idx+1, "TimeLimit")
            else:
                wrappers.insert(0, "TimeLimit")

    for wrapper in wrappers:
        wrap = getattr(mwrappers, wrapper) 
        if wrapper == "GeneralisationWrapper":
            assert generalisation_envs
            env = wrap(env, name, **arguments)
        elif wrapper == "TimeLimit":
            assert max_ep_length is not None
            env = wrap(env, max_ep_length)
        else:
            env = wrap(env)
    env.seed(seed)
    return env


def make_env(seed, **env):
    if "rware" in env["name"] and "img" in env["name"]:
        rware.image_registration()
    env = DictConfig(env)
    if env.parallel_envs:
        return _make_parallel_envs(**env, seed=seed)
    return _make_env(**env, seed=seed)
