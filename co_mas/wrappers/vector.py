import gymnasium as gym
import numpy as np
from pettingzoo.utils.env import AgentID

from co_mas.vector.vector_env import BaseVectorParallelEnvWrapper, VectorParallelEnv


class AgentStateVectorParallelEnvWrapper(BaseVectorParallelEnvWrapper):
    """
    A wrapper that return 'state' for each agent, each sub-environment must return states for each agent.
    """

    def __init__(self, env: VectorParallelEnv):
        assert all(
            hasattr(_e, "state") and hasattr(_e, "state_spaces") and hasattr(_e, "state_space") for _e in env.envs
        )
        super().__init__(env)
        self.state_spaces = self.env.envs[0].state_spaces

    def state_space(self, agent: AgentID) -> gym.Space:
        return self.state_spaces[agent]


class SyncAgentStateVectorParallelEnvWrapper(AgentStateVectorParallelEnvWrapper):
    """
    Vectorized PettingZoo Parallel environment that serially runs multiple environments and returns 'state' for each agent.
    """

    def __init__(self, env: VectorParallelEnv):
        super().__init__(env)

        self._states = {
            agent: [
                np.zeros(
                    shape=self.observation_spaces[agent][env_id].shape,
                    dtype=self.observation_spaces[agent][env_id].dtype,
                )
                for env_id in range(self.num_envs)
            ]
            for agent in self.possible_agents
        }
