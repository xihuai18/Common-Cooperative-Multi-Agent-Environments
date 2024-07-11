from typing import Any, Tuple

import pettingzoo
from pettingzoo.utils.env import AgentID


class OrderForcingParallelEnvWrapper(pettingzoo.utils.BaseParallelWrapper):
    """A wrapper that forces the order of agents in the environment"""

    def __init__(self, env: pettingzoo.ParallelEnv | pettingzoo.utils.BaseParallelWrapper):
        super().__init__(env)
        self._has_reset = False

    @property
    def has_reset(self):
        return self._has_reset

    def step(self, actions: dict[AgentID, Any]) -> Tuple:
        if not self._has_reset:
            raise RuntimeError("Environment must be reset before stepping")
        return super().step(actions)

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple:
        self._has_reset = True
        return super().reset(seed, options)


class AutoResetParallelEnvWrapper(pettingzoo.utils.BaseParallelWrapper):
    """A wrapper that automatically resets the environment when all agents are done"""

    def __init__(self, env: pettingzoo.ParallelEnv | pettingzoo.utils.BaseParallelWrapper):
        super().__init__(env)
        self._autoreset = False

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple:
        self._autoreset = False
        return super().reset(seed, options)

    def step(self, actions: dict[AgentID, Any]) -> Tuple:
        if self._autoreset:
            observation, info = self.reset()
            reward, terminated, truncated = {}, {}, {}
            for agent in self.agents:
                reward[agent] = 0
                terminated[agent] = False
                truncated[agent] = False
        else:
            observation, reward, terminated, truncated, info = super().step(actions)
        self._autoreset = len(self.agents) == 0
        return observation, reward, terminated, truncated, info
