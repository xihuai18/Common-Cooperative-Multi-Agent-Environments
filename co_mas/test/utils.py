from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from pettingzoo.utils.env import ActionType, AgentID, ObsType


def sample_action(
    agent: AgentID,
    agent_obs: ObsType,
    agent_info: Dict,
    action_space: gym.spaces.Space,
) -> ActionType:
    """
    Sample a random action.
    """
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        act_mask = agent_obs["action_mask"].astype(np.int8)
    elif "action_mask" in agent_info:
        act_mask = agent_info["action_mask"].astype(np.int8)
    else:
        act_mask = None
    if act_mask is not None:
        # return action_space.sample(act_mask)
        return np.random.choice(np.flatnonzero(act_mask))
    else:
        # return action_space.sample()
        return np.random.uniform(action_space.low, action_space.high, action_space.shape).squeeze()


def vector_sample_sample(
    agent: AgentID,
    agent_obs: Tuple[ObsType],
    agent_info: Dict[Any, Tuple],
    agent_space: gym.spaces.Tuple,
    envs_have_agent: Tuple[int],
) -> Tuple[ActionType]:
    """
    Sample random actions for a vector environment.
    """
    action = []
    for i, env_id in enumerate(envs_have_agent):
        if isinstance(agent_obs[i], dict) and "action_mask" in agent_obs[i]:
            act_mask = agent_obs[i]["action_mask"].astype(np.int8)
        elif "action_mask" in agent_info:
            act_mask = agent_info["action_mask"][i].astype(np.int8)
        else:
            act_mask = None
        if act_mask is not None:
            # action.append(agent_space[env_id].sample(act_mask))
            action.append(np.random.choice(np.flatnonzero(act_mask)))
        else:
            # action.append(agent_space[env_id].sample())
            action.append(
                np.random.uniform(
                    agent_space[env_id].low, agent_space[env_id].high, agent_space[env_id].shape
                ).squeeze()
            )
    return tuple(action)
