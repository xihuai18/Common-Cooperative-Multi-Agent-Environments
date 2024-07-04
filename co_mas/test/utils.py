import numpy as np
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv


def sample_action(
    env: ParallelEnv[AgentID, ObsType, ActionType],
    obs: dict[AgentID, ObsType],
    agent: AgentID,
    info: dict[AgentID, dict],
) -> ActionType:
    # logger.debug(info)
    # logger.debug(agent)
    agent_obs = obs[agent]
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        legal_actions = np.flatnonzero(agent_obs["action_mask"])
        if len(legal_actions) == 0:
            return 0
        return env.np_random.choice(legal_actions)
    elif "action_mask" in info[agent]:
        legal_actions = np.flatnonzero(info[agent]["action_mask"])
        if len(legal_actions) == 0:
            return 0
        return env.np_random.choice(legal_actions)
    return env.action_space(agent).sample()
