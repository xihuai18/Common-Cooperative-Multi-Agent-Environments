""" 
Modified from https://github/Farama-Foundation/SuperSuit/supersuit/vector/markov_vector_wrapper.py
"""

import re
import warnings
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import AgentID

__all__ = ["change_space", "get_indicator_map", "change_value"]


def change_space(space: gym.Space, num_indicators: int):
    if isinstance(space, Box):
        ndims = len(space.shape)
        if ndims == 1:
            pad_space = np.min(space.high) * np.ones((num_indicators,), dtype=space.dtype)
            new_low = np.concatenate([space.low, np.zeros_like(pad_space)], axis=0)
            new_high = np.concatenate([space.high, pad_space], axis=0)
            new_space = Box(low=new_low, high=new_high, dtype=space.dtype)
            return new_space
        elif ndims == 3 or ndims == 2:
            orig_low = space.low if ndims == 3 else np.expand_dims(space.low, 2)
            orig_high = space.high if ndims == 3 else np.expand_dims(space.high, 2)
            pad_space = np.min(space.high) * np.ones(orig_low.shape[:2] + (num_indicators,), dtype=space.dtype)
            new_low = np.concatenate([orig_low, np.zeros_like(pad_space)], axis=2)
            new_high = np.concatenate([orig_high, pad_space], axis=2)
            new_space = Box(low=new_low, high=new_high, dtype=space.dtype)
            return new_space
    elif isinstance(space, Discrete):
        return Discrete(space.n * num_indicators)

    raise NotImplementedError(f"agent_indicator space must be 1d, 2d, or 3d Box or Discrete, while it is {space}")


def get_indicator_map(agents: List[AgentID], type_only: bool = False) -> Dict[AgentID, int]:
    if type_only:
        assert all(
            re.match("[a-z]+_[0-9]+", agent) for agent in agents
        ), "when the `type_only` parameter is True to agent_indicator, the agent names must follow the `<type>_<n>` format"
        agent_id_map = {}
        type_idx_map = {}
        idx_num = 0
        for agent in agents:
            agent_type = agent.split("_")[0]
            if agent_type not in type_idx_map:
                type_idx_map[agent_type] = idx_num
                idx_num += 1
            agent_id_map[agent] = type_idx_map[agent_type]
        if idx_num == 1:
            warnings.warn("agent_indicator wrapper is degenerate, only one agent type; doing nothing")
        return agent_id_map
    else:
        return {agent: i for i, agent in enumerate(agents)}


def change_value(value: np.ndarray, space: gym.Space, indicator_value: float, num_indicators: int) -> np.ndarray:
    assert 0 <= indicator_value < num_indicators
    if isinstance(space, Box):
        ndims = len(space.shape)
        if ndims == 1:
            old_len = len(value)
            new_obs = np.pad(value, (0, num_indicators))
            # if all spaces are finite, use the max, otherwise use 1.0 as agent indicator
            if not np.isinf(space.high).any():
                new_obs[indicator_value + old_len] = np.max(space.high)
            else:
                new_obs[indicator_value + old_len] = 1.0

            return new_obs
        elif ndims == 3 or ndims == 2:
            value = value if ndims == 3 else np.expand_dims(value, 2)
            old_shaped3 = value.shape[2]
            new_obs = np.pad(value, [(0, 0), (0, 0), (0, num_indicators)])
            # if all spaces are finite, use the max, otherwise use 1.0 as agent indicator
            if not np.isinf(space.high).any():
                new_obs[:, :, old_shaped3 + indicator_value] = np.max(space.high)
            else:
                new_obs[:, :, old_shaped3 + indicator_value] = 1.0
            return new_obs
    elif isinstance(space, Discrete):
        return value * num_indicators + indicator_value

    raise NotImplementedError(f"agent_indicator space must be 1d, 2d, or 3d Box or Discrete, while it is {space}")
