from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterator, Sequence, Tuple

import gymnasium as gym
import numpy as np
from pettingzoo.utils.env import ActionType, AgentID, ObsType

from co_mas.env import ParallelEnv
from co_mas.vector.vector_env import VectorParallelEnv
from co_mas.wrappers import AutoResetParallelEnvWrapper


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
        self.single_observation_spaces = self.envs[0].observation_spaces
        self.single_action_spaces = self.envs[0].action_spaces
        self.possible_agents = self.envs[0].possible_agents
        self._check_spaces()

        self.observation_spaces = {
            agent: gym.spaces.Tuple([self.envs[0].observation_space(agent)] * self.num_envs)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Tuple([self.envs[0].action_space(agent)] * self.num_envs)
            for agent in self.possible_agents
        }
        self.agents = tuple(env.agents for env in self.envs)
        self.agents_old = tuple([] for _ in self.envs)
        self.envs_have_agents = defaultdict(list)
        self._update_envs_have_agents()

        self._mark_envs()
        # record which environments will autoreset
        self._autoreset_envs = np.zeros(shape=(self.num_envs,), dtype=bool)

    def _check_spaces(self) -> bool:
        """Check that each of the environments obs and action spaces are equivalent to the single obs and action space."""
        for env in self.envs:
            if not (env.observation_spaces == self.single_observation_spaces):
                raise RuntimeError(
                    f"Some environments have an observation space different from `{self.single_observation_spaces}`. "
                    "In order to batch observations, the observation spaces from all environments must be equal."
                )

            if not (env.action_spaces == self.single_action_spaces):
                raise RuntimeError(
                    f"Some environments have an action space different from `{self.single_action_spaces}`. "
                    "In order to batch actions, the action spaces from all environments must be equal."
                )

        return True

    def _mark_envs(self):
        def _mark_env(env: ParallelEnv):
            # check if the environment will autoreset
            while hasattr(env, "env"):
                if isinstance(env, AutoResetParallelEnvWrapper):
                    return False
                env = env.env
            return True

        self._need_autoreset_envs = [_mark_env(env) for env in self.envs]

    def _update_envs_have_agents(self):
        for env_id, (_agents_in_env, _agents_in_env_old) in enumerate(zip(self.agents, self.agents_old)):
            add_agents = set(_agents_in_env) - set(_agents_in_env_old)
            remove_agents = set(_agents_in_env_old) - set(_agents_in_env)
            for agent in add_agents:
                self.envs_have_agents[agent].append(env_id)
            for agent in remove_agents:
                self.envs_have_agents[agent].remove(env_id)

    def reset(
        self, seed: int | list[int] | None = None, options: dict | None = None
    ) -> Tuple[Dict[AgentID, Tuple[ObsType]], Dict[AgentID, Dict]]:
        if seed is None:
            seed = [None] * self.num_envs
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]

        assert len(seed) == self.num_envs, "The number of seeds must match the number of environments."

        observation = {agent: [] for agent in self.possible_agents}
        env_infos = []

        for env, seed in zip(self.envs, seed):
            obs, info = env.reset(seed=seed, options=options)
            for agent in self.possible_agents:
                if agent in obs:
                    observation[agent].append(obs[agent])
            env_infos.append(info)
        info = self._merge_infos(env_infos)

        self._construct_batch_result_in_place(observation)

        self.agents = tuple(env.agents for env in self.envs)
        self.agents_old = tuple([] for _ in self.envs)
        self.envs_have_agents = defaultdict(list)
        self._update_envs_have_agents()

        self._autoreset_envs = np.zeros(shape=(self.num_envs,), dtype=bool)

        return observation, info

    def step(self, actions: Dict[AgentID, Tuple[ActionType]]) -> Tuple[
        Dict[AgentID, Tuple[ObsType]],
        Dict[AgentID, Tuple[float]],
        Dict[AgentID, Tuple[bool]],
        Dict[AgentID, Tuple[bool]],
        Dict[AgentID, Dict],
    ]:
        observation = {agent: [] for agent in self.possible_agents}
        reward = {agent: [] for agent in self.possible_agents}
        termination = {agent: [] for agent in self.possible_agents}
        truncation = {agent: [] for agent in self.possible_agents}
        env_infos = []

        env_actions = [{} for _ in self.envs]
        for agent, agent_actions in actions.items():
            for i, env_id in enumerate(self.envs_have_agents[agent]):
                env_actions[env_id][agent] = agent_actions[i]
        for env_id, env, env_acts in enumerate(zip(self.envs, env_actions)):
            if self._autoreset_envs[env_id]:
                obs, info = env.reset()
                for agent in env.agents:
                    observation[agent].append(obs[agent])
                    reward[agent].append(0)
                    termination[agent].append(False)
                    truncation[agent].append(False)
                env_infos.append(info)
            else:
                obs, rew, term, trunc, info = env.step(env_acts)
                for agent in env.agents:
                    observation[agent].append(obs[agent])
                    reward[agent].append(rew[agent])
                    termination[agent].append(term[agent])
                    truncation[agent].append(trunc[agent])
                env_infos.append(info)

        self._construct_batch_result_in_place(observation)
        self._construct_batch_result_in_place(reward)
        self._construct_batch_result_in_place(termination)
        self._construct_batch_result_in_place(truncation)
        info = self._merge_infos(env_infos)

        self._autoreset_envs = np.array([len(env.agents) == 0 for env in self.envs], dtype=bool)
        self._autoreset_envs = np.logical_and(self._autoreset_envs, self._need_autoreset_envs)

        self.agents_old = self.agents
        self.agents = tuple(env.agents for env in self.envs)
        self._update_envs_have_agents()

        return observation, reward, termination, truncation, info

    def close(self):
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True

    def state(self) -> Tuple | Dict[AgentID, Tuple]:
        pass
        # TODO: define `self.state_space` and `self.single_state_space`, otherwise raise NotImplementedError or RuntimeError to warn using `AgentStateVectorParallelEnvWrapper`

    @property
    def num_agents(self) -> Tuple[int]:
        return tuple(env.num_agents for env in self.envs)

    def __getattr__(self, name: str) -> Tuple:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return (getattr(env, name) for env in self.envs)
