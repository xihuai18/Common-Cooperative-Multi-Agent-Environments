from typing import Dict

import gymnasium as gym
from pettingzoo.utils.env import AgentID, ObsType

from co_mas.vector import AsyncVectorParallelEnv, SyncVectorParallelEnv
from co_mas.vector.utils import dict_space_from_dict_single_spaces
from co_mas.vector.vector_env import BaseVectorParallelEnvWrapper, EnvID

__all__ = [
    "AgentStateVectorParallelEnvWrapper",
    "SyncAgentStateVectorParallelEnvWrapper",
    "AsyncAgentStateVectorParallelEnvWrapper",
]


class AgentStateVectorParallelEnvWrapper(BaseVectorParallelEnvWrapper):
    """
    A wrapper that return 'state' for each agent, each sub-environment must return states for each agent.
    """

    single_state_spaces: Dict[AgentID, gym.Space]
    state_spaces: Dict[AgentID, gym.spaces.Dict]

    def state_space(self, agent: AgentID) -> gym.spaces.Dict:
        """
        Return the state space for the given agent in each sub-environments.
        """
        return self.state_spaces[agent]

    def single_state_space(self, agent: AgentID) -> gym.Space:
        """
        Return the state space for the given agent in a sub-environment.
        """
        return self.single_state_spaces[agent]

    def state(self) -> Dict[AgentID, gym.spaces.Tuple]:
        """
        Return the state of all sub environments.
        """
        raise NotImplementedError


class SyncAgentStateVectorParallelEnvWrapper(AgentStateVectorParallelEnvWrapper):
    """
    Vectorized PettingZoo Parallel environment that serially runs multiple environments and returns 'state' for each agent.
    """

    def __init__(self, env: SyncVectorParallelEnv | BaseVectorParallelEnvWrapper):
        assert all(
            hasattr(_e, "state") and hasattr(_e, "state_spaces") and hasattr(_e, "state_space") for _e in env.envs
        ), "Sub-environments should have `state`, `state_spaces` and `state_space` attributes."
        super().__init__(env)
        self.single_state_spaces = self.env.envs[0].state_spaces
        self.state_spaces = dict_space_from_dict_single_spaces(
            self.single_state_spaces, self.possible_agents, self.env.env_ids
        )

    def state(self) -> Dict[AgentID, Dict[EnvID, ObsType]]:
        state = {agent: {} for agent in self.possible_agents}
        for env_id, env in zip(self.env.env_ids, self.env.envs):
            env_state = env.state()
            if self.env.debug:
                self.env._check_containing_agents(self.env.agents_old[env_id], env_state)
            for agent in self.env.agents_old[env_id]:
                state[agent][env_id] = env_state[agent]
        self.env.construct_batch_result_in_place(state)

        return state


class AsyncAgentStateVectorParallelEnvWrapper(AgentStateVectorParallelEnvWrapper):
    """
    Vectorized PettingZoo Parallel environment that runs multiple environments asynchronously and returns 'state' for each agent.
    """

    def __init__(self, env: AsyncVectorParallelEnv | BaseVectorParallelEnvWrapper):
        assert (
            hasattr(env.dummy_env, "state")
            and hasattr(env.dummy_env, "state_space")
            and hasattr(env.dummy_env, "state_spaces")
        ), "Sub-environments should have `state`, `state_spaces` and `state_space` attributes."
        super().__init__(env)
        self.single_state_spaces = self.env._single_state_spaces
        self.state_spaces = self.env._state_spaces
        self.agent_state = self.env._agent_state

    def state(self) -> Dict[AgentID, Dict[EnvID, ObsType]]:
        return self.agent_state()
