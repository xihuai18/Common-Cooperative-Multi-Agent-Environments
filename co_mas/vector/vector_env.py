"""
High performance vectorized multi-agent environment for PettingZoo Parallel environments. 

Modified from https://gymnasium.farama.org/api/vector/#observation_space.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from typing import Any, Dict, Generic, List, Tuple

import gymnasium as gym
from pettingzoo.utils.env import ActionType, AgentID, ObsType

__all__ = ["VectorParallelEnv", "BaseVectorParallelEnvWrapper"]


class VectorParallelEnv(Generic[AgentID, ObsType, ActionType]):
    """
    High performance vectorized multi-agent environment for PettingZoo Parallel environments.
    All sub-environments must have the same observation, state and action spaces.
    NOTE: all sub-environments will be reset automatically if there is not agent in the environment, except for those wrapped by `AutoResetParallelEnvWrapper`.

    We use Tuple to construct the spaces of VectorParallelEnv, each element of the Tuple is a gym.spaces.Space from a sub-environment.

    `envs_have_agents` and `envs_have_agent` are used to record the sub-environment indices that have the agent, which is useful for constructing actions.

    # TODO: doc for input and output
    """

    metadata: dict[str, Any] = {}

    num_envs: int

    possible_agents: list[AgentID]
    agents: Tuple[List[AgentID]]
    _envs_have_agents: Dict[AgentID, Tuple[int]] = defaultdict(list)

    # Spaces for each agents and all sub-environment
    observation_spaces: dict[AgentID, gym.spaces.Tuple]
    action_spaces: dict[AgentID, gym.spaces.Tuple]

    # Spaces for each agents and one sub-environment
    single_observation_spaces: dict[AgentID, gym.spaces.Space]
    single_action_spaces: dict[AgentID, gym.spaces.Space]

    # if auto_need_autoreset_envs[i] == True, manually reset it, otherwise env `i` will be reset automatically.
    _need_autoreset_envs: List[bool]

    closed: bool = False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Dict[AgentID, Tuple[ObsType]], Dict[AgentID, Dict]]:
        """
        Reset all parallel environments and return a batch of initial observations and info.
        """
        raise NotImplementedError

    def step(self, actions: Dict[AgentID, Tuple[ActionType]]) -> Tuple[
        Dict[AgentID, Tuple[ObsType]],
        Dict[AgentID, Tuple[float]],
        Dict[AgentID, Tuple[bool]],
        Dict[AgentID, Tuple[bool]],
        Dict[AgentID, Dict],
    ]:
        """
        Step all parallel environments with the given actions and return a batch of results.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close all parallel environments.
        """
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

    def state(self) -> Tuple:
        """
        Return the state of all parallel environments.
        """
        raise NotImplementedError

    def single_observation_space(self, agent: AgentID) -> gym.Space:
        """
        Return the observation space for the given agent in a sub-environment.
        """
        return self.single_observation_spaces[agent]

    def observation_space(self, agent: AgentID) -> gym.Space:
        """
        Return the observation space for the given agent.
        """
        return self.observation_spaces[agent]

    def single_action_space(self, agent: AgentID) -> gym.Space:
        """
        Return the action space for the given agent in a sub-environment.
        """
        return self.single_action_spaces[agent]

    def action_space(self, agent: AgentID) -> gym.Space:
        """
        Return the action space for the given agent.
        """
        return self.action_spaces[agent]

    @functools.lru_cache()
    def envs_have_agent(self, agent: AgentID) -> Tuple[int]:
        """
        return the sub environment indices that have the agent
        """
        return tuple(self._envs_have_agents[agent])

    @functools.lru_cache()
    def envs_have_agents(self) -> Dict[AgentID, Tuple[int]]:
        """
        return the sub environment indices that have the agent
        """
        return {agent: tuple(self._envs_have_agents[agent]) for agent in self._envs_have_agents}

    def _merge_infos(self, env_infos: Tuple[Dict[AgentID, Dict]]) -> Dict[AgentID, Dict]:
        """
        Merge env_infos into vector_info.
        """
        vector_info = {}

        def _merge(_infos: Tuple[Dict], _key: Any) -> Dict:
            """
            Merge values in _env_infos for _key into a tuple.
            """
            _merged_info = {}
            _dummy_info = _infos[0]
            if isinstance(_dummy_info[_key], dict):
                for _k in _dummy_info[_key].keys():
                    _merged_info[_k] = _merge(_infos=tuple(_info[_key] for _info in _infos), _key=_k)
            else:
                _merged_info = tuple([_info[_key] for _info in _infos])
            return _merged_info

        for agent in self.possible_agents:
            infos_for_agent = tuple(_env_info[agent] for _env_info in env_infos if agent in _env_info)
            if len(infos_for_agent) <= 0:
                continue
            vector_info[agent] = {}
            dummy_info = infos_for_agent[0]
            for key in dummy_info.keys():
                vector_info[agent][key] = _merge(infos_for_agent, _key=key)

        return vector_info

    def _construct_batch_result_in_place(self, result: Dict[AgentID, List]) -> Dict[AgentID, Tuple]:
        """
        remove empty agent results and convert them to tuple in-place.
        """
        for agent in self.possible_agents:
            if len(result[agent]) <= 0:
                result.pop(agent)
            else:
                result[agent] = tuple(result[agent])

    @property
    def num_agents(self) -> Tuple[int]:
        """return the number of agents in each sub-environment."""
        raise NotImplementedError

    @property
    def max_num_agents(self) -> int:
        """return the maximum number of agents in all sub-environments."""
        return len(self.possible_agents)


class BaseVectorParallelEnvWrapper(VectorParallelEnv[AgentID, ObsType, ActionType]):
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
