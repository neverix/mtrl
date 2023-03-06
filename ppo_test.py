import pickle
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from gym.spaces import Dict as DictOAI, Box, Discrete, MultiDiscrete
from gym.wrappers import Monitor, TimeLimit, ResizeObservation, FrameStack
from minetester import Minetest
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from typing import Optional, Dict, Any, List, Tuple
from stable_baselines3.common.distributions import (
    Distribution,
    MultiCategoricalDistribution,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.type_aliases import Schedule
import torch as th
import torch
import gym
import gc
import sys
# TODO create package for VPT
if "./video-pre-training" not in sys.path:
    sys.path.append("./video-pre-training")
from agent import MineRLAgent


MINERL_TO_MINETEST_ACTIONS = {
    "ESC": "ESC",  # no VPT action
    "attack": "DIG",
    "back": "BACKWARD",
    "camera": "MOUSE",
    "drop": "DROP",
    "forward": "FORWARD",
    "hotbar.1": "SLOT_1",
    "hotbar.2": "SLOT_2",
    "hotbar.3": "SLOT_3",
    "hotbar.4": "SLOT_4",
    "hotbar.5": "SLOT_5",
    "hotbar.6": "SLOT_6",
    "hotbar.7": "SLOT_7",
    "hotbar.8": "SLOT_8",
    "hotbar.9": "SLOT_1",
    "inventory": "INVENTORY",
    "jump": "JUMP",
    "left": "LEFT",
    "pickItem": "MIDDLE",  # no VPT action
    "right": "RIGHT",
    "sneak": "SNEAK",
    "sprint": None,
    "swapHands": None,  # no VPT action
    "use": "PLACE",
}


def minerl_to_minetest_action(minerl_action, minetest_env):
    """
    Wrapper that makes VPT models work with Minetest.
    Unused for now.
    """
    minetest_action = {}
    for minerl_key, minetest_key in MINERL_TO_MINETEST_ACTIONS.items():
        if minetest_key and minerl_key in minerl_action:
            if minetest_key != "MOUSE":
                minetest_action[minetest_key] = int(minerl_action[minerl_key][0])
            else:
                # TODO this translation of the camera action maybe wrong
                camera_action = minerl_action[minerl_key][0]
                mouse_action = [0, 0]
                mouse_action[0] = round(
                    0.5
                    * minetest_env.display_size[0]
                    * camera_action[0]
                    / minetest_env.fov_x,
                )
                mouse_action[1] = round(
                    0.5
                    * minetest_env.display_size[1]
                    * camera_action[1]
                    / minetest_env.fov_y,
                )
                print(f"Camera action {camera_action}, mouse action {mouse_action}")
                minetest_action[minetest_key] = mouse_action
    minetest_action["HOTBAR_NEXT"] = minetest_action["HOTBAR_PREV"] = 0
    minetest_action["ESC"] = minetest_action["MIDDLE"] = 0
    return minetest_action


def minetest_to_minerl_obs(minetest_obs):
    return {"pov": minetest_obs}


def mu_law(x):
    x = x * 2 - 1
    return x


class FrameSkip(gym.Wrapper):
    """Copy-pasted FrameSkip from Gym (why did they remove this)"""

    def __init__(self, env, frame_skip=4):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, action):
        result = None
        for _ in range(self.frame_skip):
            result = self.env.step(action)
        return result


class DiscreteActions(gym.ActionWrapper):
    """Turn Mine(test|RL)'s Dict of Boxes and Discretes into a MultiDiscrete. Hacky but works (?)"""
    def __init__(self, env, discretes=11):
        super().__init__(env)
        self.discretes = discretes
        sizes = []
        self.vals = []
        for i, v in env.action_space.spaces.items():
            self.vals.append(len(sizes))
            if isinstance(v, Discrete):
                sizes.append(v.n)
            elif isinstance(v, Box):
                for _ in v.low:
                    sizes.append(discretes)

        self.action_space = MultiDiscrete(sizes)  # TODO

    def action(self, act):
        return {k: (np.asarray(act[i:i + len(v.low)]) / self.discretes * (v.high - v.low) + v.low
                    if isinstance(v, Box) else act[i]).astype(v.dtype)
                for i, (k, v) in zip(self.vals, self.env.action_space.spaces.items())}


class GetPOV(gym.ObservationWrapper):
    """Creates an observation space and provides a zero observation."""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(64, 64, 3,), low=0, high=255)

    def observation(self, obs):
        if obs.shape != self.observation_space.shape:
            return np.zeros(self.observation_space.shape)
        return obs


def make_env(
        minetest_path: str,
        rank: int,
        seed: int = 0,
        max_steps: int = 1e9,
        env_kwargs: Optional[Dict[str, Any]] = None,
):
    env_kwargs = env_kwargs or {}

    def _init():
        env = Minetest(
            # Make sure that each Minetest instance has different server and client ports
            env_port=5555 + rank,
            server_port=30000 + rank,
            seed=None,  # Create worlds randomly
            world_dir=f"worlds/myworld{rank}",  # Store worlds locally
            minetest_executable=None,
            xvfb_headless=False,  # TODO Doesn't work on my machine
            config_path="minetest.conf",  # Needs to be concrete, errors out otherwise
            **env_kwargs,
        )
        env.reset_world = True
        env = GetPOV(env)
        env = ResizeObservation(env, 64)
        env = FrameSkip(env)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = DiscreteActions(env)
        return env

    return _init


if __name__ == "__main__":
    # Env settings
    seed = 42
    max_steps = 1000
    env_kwargs = {"display_size": (1024, 600), "fov": 72}

    # Create a vectorized environment
    num_envs = 2  # Number of envs to use (<= number of avail. cpus)
    vec_env_cls = SubprocVecEnv
    venv = vec_env_cls(
        [
            make_env(minetest_path=minetest_path, rank=i, seed=seed, max_steps=max_steps, env_kwargs=env_kwargs)
            for i in range(num_envs)
        ],
    )

    ppo = PPO("CnnPolicy", venv, verbose=1, batch_size=2, n_steps=1)
    ppo.learn(total_timesteps=25000)

