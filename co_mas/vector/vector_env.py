"""
High performance vectorized multi-agent environment for PettingZoo Parallel environments. 

Modified from https://gymnasium.farama.org/api/vector/#observation_space.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from typing import Any, Generic, TypeVar

import gymnasium as gym
from pettingzoo.utils.env import ActionType, AgentID, ObsType

EnvID = TypeVar("EnvID")

__all__ = ["VectorParallelEnv", "BaseVectorParallelEnvWrapper", "EnvID"]


class VectorParallelEnv(Generic[EnvID, AgentID, ObsType, ActionType]):
    """
    High performance vectorized multi-agent environment for PettingZoo Parallel environments.
    All sub-environments must have the same observation, state and action spaces.
    NOTE: all sub-environments will be reset automatically if there is not agent in the environment, except for those wrapped by `AutoResetParallelEnvWrapper`.

    We use gymnasium.spaces.Dict to construct the spaces of VectorParallelEnv, each item pair of the Dict is an env_id and a gym.spaces.Space from a sub-environment.

    `envs_have_agents` and `envs_have_agent` are used to record the sub-environment indices that have the agent, which is useful for constructing actions.

    For `reset` and `step` methods, `observations`, `rewards`, `terminations`, `truncations` and `infos` are all Dicts of Dicts, where the first Dict is the agent_id and the second Dict is the env_id.

    For the `state` method, the return value is a Dict of state arrays, where the key is the env_id. If you want it to return agent specific states, you can use the provided `SyncAgentStateVectorParallelEnvWrapper` or `AsyncAgentStateVectorParallelEnvWrapper` wrappers.

    We provide `SyncVectorParallelEnv` and `AsyncVectorParallelEnv` for serial and parallel environments, respectively. For detailed usage, please refer to the examples in the `tests` directory or the documentation of the corresponding classes.
    """

    metadata: dict[str, Any] = {}

    num_envs: int

    possible_agents: tuple[AgentID]
    agents: dict[EnvID, tuple[AgentID]]
    agents_old: dict[EnvID, tuple[AgentID]]
    env_ids: tuple[EnvID]
    _envs_have_agents: dict[AgentID, list[EnvID]] = defaultdict(list)

    # Spaces for each agents and all sub-environment
    observation_spaces: dict[AgentID, gym.spaces.Dict]
    action_spaces: dict[AgentID, gym.spaces.Dict]

    # Spaces for each agents and one sub-environment
    single_observation_spaces: dict[AgentID, gym.spaces.Space]
    single_action_spaces: dict[AgentID, gym.spaces.Space]

    closed: bool = False

    def reset(
        self, seed: int | list[int] | dict[EnvID, int] | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, dict[EnvID, ObsType]], dict[AgentID, dict[EnvID, dict]]]:
        """
        Reset all parallel environments and return a batch of initial observations and info.
        """
        raise NotImplementedError

    def step(self, actions: dict[AgentID, dict[EnvID, ActionType]]) -> tuple[
        dict[AgentID, dict[EnvID, ObsType]],
        dict[AgentID, dict[EnvID, float]],
        dict[AgentID, dict[EnvID, bool]],
        dict[AgentID, dict[EnvID, bool]],
        dict[AgentID, dict[EnvID, dict]],
    ]:
        """
        Step all parallel environments with the given actions and return a batch of results.
        """
        raise NotImplementedError

    def close(self, **kwargs) -> None:
        """
        Close all parallel environments.
        It calls `close_extras` and set `closed` as `True`.
        """
        if self.closed:
            return
        self.close_extras(**kwargs)
        self.closed = True

    def close_extras(self, **kwargs) -> None:
        """Clean up the extra resources e.g. beyond what's in this base class."""
        raise NotImplementedError

    @property
    def unwrapped(self) -> VectorParallelEnv:
        return self

    def __del__(self):
        """Forcing the environment to close."""
        self.close()

    def __exit__(self, *args: Any) -> bool:
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False

    def __repr__(self) -> str:
        """Returns a string representation of the vector environment."""

        return f"VectorParallelEnv(num_envs={self.num_envs})"

    def state(self) -> dict[EnvID, ObsType]:
        """
        Return the state of all parallel environments.
        """
        raise NotImplementedError

    @functools.lru_cache
    def single_observation_space(self, agent: AgentID) -> gym.Space:
        """
        Return the observation space for the given agent in a sub-environment.
        """
        return self.single_observation_spaces[agent]

    @functools.lru_cache
    def observation_space(self, agent: AgentID) -> gym.Space:
        """
        Return the observation space for the given agent.
        """
        return self.observation_spaces[agent]

    @functools.lru_cache
    def single_action_space(self, agent: AgentID) -> gym.Space:
        """
        Return the action space for the given agent in a sub-environment.
        """
        return self.single_action_spaces[agent]

    @functools.lru_cache
    def action_space(self, agent: AgentID) -> gym.Space:
        """
        Return the action space for the given agent.
        """
        return self.action_spaces[agent]

    def envs_have_agent(self, agent: AgentID) -> tuple[EnvID]:
        """
        return the sub environment indices that have the agent
        """
        return tuple(self._envs_have_agents[agent])

    @property
    def envs_have_agents(self) -> dict[AgentID, tuple[EnvID]]:
        """
        return the sub environment indices that have the agent
        """
        return {agent: tuple(self._envs_have_agents[agent]) for agent in self._envs_have_agents.keys()}

    def add_info_in_place(
        self, vector_info: dict[AgentID, dict[Any, dict[EnvID, Any]]], env_info: dict[AgentID, dict], env_id: EnvID
    ):
        """
        add an env_info for env_id into the vector_info.
        """

        def _merge(_vector_info: dict[Any, dict[EnvID, Any]], _env_info: dict):
            """
            Merge values in _env_info into _vector_info.
            """
            for k, v in _env_info.items():
                if isinstance(v, dict):
                    _merge(_vector_info[k], v)
                else:
                    if k not in _vector_info:
                        _vector_info[k] = {env_id: v}
                    else:
                        _vector_info[k][env_id] = v

        for agent in self.possible_agents:
            if agent in env_info:
                _merge(vector_info[agent], env_info[agent])

    def construct_batch_result_in_place(
        self, result: dict[AgentID, dict[EnvID, Any]]
    ) -> dict[AgentID, dict[EnvID, Any]]:
        """
        remove empty agent results.
        """
        for agent in self.possible_agents:
            if len(result[agent]) <= 0:
                result.pop(agent)

    @property
    def num_agents(self) -> dict[EnvID, tuple[int]]:
        """return the number of agents in each sub-environment."""
        raise NotImplementedError

    @property
    def max_num_agents(self) -> int:
        r"""return the maximum number of agents in all\ sub-environments."""
        return len(self.possible_agents)

    def _update_envs_have_agents(self):
        for env_id in self.env_ids:
            _agents_in_env = set(self.agents[env_id])
            _agents_in_env_old = set(self.agents_old[env_id])
            add_agents = _agents_in_env - _agents_in_env_old
            remove_agents = _agents_in_env_old - _agents_in_env
            for agent in add_agents:
                self._envs_have_agents[agent].append(env_id)
            for agent in remove_agents:
                self._envs_have_agents[agent].remove(env_id)

    def _check_containing_agents(self, agents: tuple[AgentID], env_results: dict[AgentID, Any]) -> bool:
        assert set(agents) == set(
            env_results.keys()
        ), f"The agents in the environment {sorted(list(agents))} should be the same as the agents in the results {sorted(list(env_results.keys()))}."


class BaseVectorParallelEnvWrapper(Generic[EnvID, AgentID, ObsType, ActionType]):
    """
    Base wrapper for VectorParallelEnv.
    """

    def __init__(self, env: VectorParallelEnv[AgentID, ObsType, ActionType]):
        super().__init__()
        self.env = env

    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.env})"
