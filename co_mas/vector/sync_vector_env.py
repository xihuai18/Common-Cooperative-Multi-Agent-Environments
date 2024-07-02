from __future__ import annotations

from typing import Callable, Iterator, Sequence

import gymnasium as gym
import numpy as np

from co_mas.env import ParallelEnv
from co_mas.vector.vector_env import VectorParallelEnv


class SyncVectorParallelEnv(VectorParallelEnv):
    """
    Vectorized PettingZoo Parallel environment that serially runs multiple environments.
    """

    def __init__(self, env_fns: Iterator[Callable[[], ParallelEnv]] | Sequence[Callable[[], ParallelEnv]]):

        self.env_fns = env_fns
        # Initialise all sub-environments
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)

        self.metadata = self.envs[0].metadata
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.possible_agents = self.envs[0].possible_agents
        self._check_spaces()

        self.observation_spaces = {
            agent: gym.spaces.Tuple([self.envs[0].observation_space] * self.num_envs) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Tuple([self.envs[0].action_space] * self.num_envs) for agent in self.possible_agents
        }
        self.agents = tuple(env.agents for env in self.envs)

        self._mark_envs()
        self._autoreset_envs = np.zeros(shape=(self.num_envs,), dtype=bool)

        self._observations = {
            agent: [
                np.zeros(
                    shape=self.observation_spaces[agent][env_id].shape,
                    dtype=self.observation_spaces[agent][env_id].dtype,
                )
                for env_id in range(self.num_envs)
            ]
            for agent in self.possible_agents
        }
        self._rewards = {agent: np.zeros(shape=(self.num_envs,), dtype=float) for agent in self.possible_agents}
        self._terminations = {agent: np.zeros(shape=(self.num_envs,), dtype=bool) for agent in self.possible_agents}
        self._truncation = {agent: np.zeros(shape=(self.num_envs,), dtype=bool) for agent in self.possible_agents}

    def _check_spaces(self) -> bool:
        """Check that each of the environments obs and action spaces are equivalent to the single obs and action space."""
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    f"Some environments have an observation space different from `{self.single_observation_space}`. "
                    "In order to batch observations, the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    f"Some environments have an action space different from `{self.single_action_space}`. "
                    "In order to batch actions, the action spaces from all environments must be equal."
                )

        return True
