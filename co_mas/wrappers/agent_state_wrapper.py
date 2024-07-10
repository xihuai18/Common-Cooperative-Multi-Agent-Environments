import functools
from typing import Dict

import gymnasium as gym
import pettingzoo
from pettingzoo.utils.env import ActionType, AgentID, ObsType

import co_mas.wrappers.utils.agent_indicator as agent_indicator


class AgentStateParallelEnvWrapper(pettingzoo.utils.BaseParallelWrapper):
    """A wrapper that return 'state' for each agent"""

    def __init__(
        self, env: pettingzoo.utils.ParallelEnv | pettingzoo.utils.BaseParallelWrapper, type_only: bool = False
    ):
        assert hasattr(env, "state_space") and hasattr(
            env, "state"
        ), "AgentStateWrapper requires the environment to have `state_space` and a `state` attributes"
        assert not hasattr(
            env, "state_spaces"
        ), "AgentStateWrapper requires the environment to not have `state_spaces` attribute"
        super().__init__(env)
        self.state_spaces = gym.spaces.Dict(
            {
                agent: agent_indicator.change_space(self.env.state_space, len(self.env.possible_agents))
                for agent in self.env.possible_agents
            }
        )
        self.type_only = type_only
        self.agents_old = None

    @functools.lru_cache()
    def state_space(self, agent: AgentID) -> gym.Space:
        return self.state_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        ret = self.env.reset(seed, options)
        self.agents_old = self.env.agents
        return ret

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        self.agents_old = self.env.agents
        ret = self.env.step(actions)
        return ret

    def state(self) -> Dict:
        state = self.env.state()
        num_agents = len(self.agents_old)
        indicator_map = agent_indicator.get_indicator_map(self.agents_old, type_only=self.type_only)
        return {
            agent: agent_indicator.change_value(state, self.state_space(agent), indicator_map[agent], num_agents)
            for agent in self.agents_old
        }
