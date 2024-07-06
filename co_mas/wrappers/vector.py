from typing import Dict

import gymnasium as gym
from pettingzoo.utils.env import AgentID, ObsType

from co_mas.vector.vector_env import (
    BaseVectorParallelEnvWrapper,
    EnvID,
    VectorParallelEnv,
)


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

    def __init__(self, env: VectorParallelEnv):
        assert all(
            hasattr(_e, "state") and hasattr(_e, "state_spaces") and hasattr(_e, "state_space") for _e in env.envs
        )
        super().__init__(env)
        self.single_state_spaces = self.env.envs[0].state_spaces
        self.state_spaces = {
            agent: gym.spaces.Dict({env_id: self.single_state_spaces[agent] for env_id in self.env.env_ids})
            for agent in self.possible_agents
        }

    def state(self) -> Dict[AgentID, Dict[EnvID, ObsType]]:
        state = {agent: {} for agent in self.possible_agents}
        for env_id, env in zip(self.env.env_ids, self.env.envs):
            env_state = env.state()
            for agent in self.possible_agents:
                if agent in env_state:
                    state[agent][env_id] = env_state[agent]
        self.env.construct_batch_result_in_place(state)

        return state
