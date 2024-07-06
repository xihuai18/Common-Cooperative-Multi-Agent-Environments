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
        if isinstance(action_space, gym.spaces.Discrete):
            return np.random.choice(action_space.n)
        else:
            return np.random.uniform(action_space.low, action_space.high, action_space.shape).squeeze()


def vector_sample_sample(
    agent: str,
    agent_obs: Dict[str, np.ndarray],
    agent_info: Dict[str, Dict],
    agent_space: gym.spaces.Dict,
    envs_have_agent: Tuple[str],
) -> Dict[str, Any]:
    """
    Sample random actions for a vector environment.
    """
    action = {}
    for env_id in envs_have_agent:
        if isinstance(agent_obs[env_id], dict) and "action_mask" in agent_obs[env_id]:
            act_mask = agent_obs[env_id]["action_mask"].astype(np.int8)
        elif "action_mask" in agent_info:
            act_mask = agent_info["action_mask"][env_id].astype(np.int8)
        else:
            act_mask = None
        if act_mask is not None:
            # action.append(agent_space[env_id].sample(act_mask))
            action[env_id] = np.random.choice(np.flatnonzero(act_mask))
        else:
            if isinstance(agent_space[env_id], gym.spaces.Discrete):
                # action.append(agent_space[env_id].sample())
                action[env_id] = np.random.choice(agent_space[env_id].n)
            else:
                # action.append(agent_space[env_id].sample())
                action[env_id] = np.random.uniform(
                    agent_space[env_id].low, agent_space[env_id].high, agent_space[env_id].shape
                ).squeeze()
    return action
