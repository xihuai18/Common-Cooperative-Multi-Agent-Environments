from typing import Dict

import gymnasium as gym
import pettingzoo
from pettingzoo.utils.env import AgentID

import co_mas.wrappers.utils.agent_indicator as agent_indicator


class AgentStateParallelEnvWrapper(pettingzoo.utils.BaseParallelWrapper):
    """A wrapper that return 'state' for each agent"""

    def __init__(self, env: pettingzoo.utils.ParallelEnv, type_only: bool = False):
        assert hasattr(env, "state_space") and hasattr(
            env, "state"
        ), "AgentStateWrapper requires the environment to have `state_space` and a `state` attributes"
        super().__init__(env)
        self.state_spaces = (
            self.env.state_space
            if isinstance(self.env.state_space, Dict)
            else {
                agent: agent_indicator.change_space(self.env.state_space, len(self.env.agents))
                for agent in self.env.agents
            }
        )
        self.type_only = type_only

    def state_space(self, agent: AgentID) -> gym.Space:
        return self.state_spaces[agent]

    def state(self) -> Dict:
        state = self.env.state()
        if isinstance(state, Dict):
            return state
        else:
            num_agents = len(self.env.agents)
            indicator_map = agent_indicator.get_indicator_map(self.env.agents, type_only=self.type_only)
            return {
                agent: agent_indicator.change_value(state, self.state_space(agent), indicator_map[agent], num_agents)
                for agent in self.env.agents
            }
