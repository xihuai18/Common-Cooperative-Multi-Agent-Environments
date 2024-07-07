from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, Iterator, Sequence, Tuple

import gymnasium as gym
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

from co_mas.vector.vector_env import EnvID, VectorParallelEnv
from co_mas.wrappers import AutoResetParallelEnvWrapper


class SyncVectorParallelEnv(VectorParallelEnv):
    """
    Vectorized PettingZoo Parallel environment that serially runs multiple environments.

    Example:
        >>> from gfootball import gfootball_pettingzoo_v1
        >>> from co_mas.vector import SyncVectorParallelEnv
        >>> def env_gfootball_fn():
        ...     return gfootball_pettingzoo_v1.parallel_env("academy_3_vs_1_with_keeper", number_of_left_players_agent_controls=2)
        ...
        >>> sync_vec_env = SyncVectorParallelEnv([env_gfootball_fn for _ in range(2)])
        >>> sync_vec_env
        SyncVectorParallelEnv(num_envs=2)
        >>> obs, info = sync_vec_env.reset(seed=42)
        >>> from pprint import pprint
        >>> pprint(obs)
        {'player_0': {
            'env_0': array([-1.0110294 , -0.        ,  0.        , -0.        ,  0.        ,
                            0.        ,  1.6176469 ,  0.        ,  1.71875   ,  0.20325357,
                            1.71875   , -0.20325357,  2.0220587 ,  0.        ,  1.7693014 ,
                            0.        ,  1.6310294 ,  0.        ,  0.6066176 , -0.        ,
                            0.7077206 ,  0.20325357,  0.7077206 , -0.20325357,  0.        ,
                            -0.        ,  0.        , -0.        ,  0.        , -0.        ,
                            1.0110294 ,  0.        ,  0.75827205,  0.        , -0.        ,
                            0.        , -0.        ,  0.        ,  0.62      , -0.        ,
                            0.11061639, -0.        ,  0.        ,  0.00616395,  1.        ,
                            0.        ,  0.        ,  1.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
                            0.        ,  0.        ,  0.        ], dtype=float32),
            'env_1': array([-1.0110294 , -0.        ,  0.        , -0.        ,  0.        ,
                            0.        ,  1.6176469 ,  0.        ,  1.71875   ,  0.20325357,
                            1.71875   , -0.20325357,  2.0220587 ,  0.        ,  1.7693014 ,
                            0.        ,  1.6310294 ,  0.        ,  0.6066176 , -0.        ,
                            0.7077206 ,  0.20325357,  0.7077206 , -0.20325357,  0.        ,
                            -0.        ,  0.        , -0.        ,  0.        , -0.        ,
                            1.0110294 ,  0.        ,  0.75827205,  0.        , -0.        ,
                            0.        , -0.        ,  0.        ,  0.62      , -0.        ,
                            0.11061639, -0.        ,  0.        ,  0.00616395,  1.        ,
                            0.        ,  0.        ,  1.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
                            0.        ,  0.        ,  0.        ], dtype=float32)},
        ...}
        >>> pprint(info)
        {'player_0': {'action_masks': {
                        'env_0': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]),
                        'env_1': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0])}},
        ...}
        >>> state = sync_vec_env.state()
        >>> pprint(state)
        {'env_0': {
            'player_0': array([
                -1.0110294 , -0.        ,  0.6066176 , -0.        ,  0.7077206 ,
                0.20325357,  0.7077206 , -0.20325357,  0.        , -0.        ,
                0.        , -0.        ,  0.        , -0.        ,  0.        ,
                -0.        ,  1.0110294 ,  0.        ,  0.75827205,  0.        ,
                -0.        ,  0.        , -0.        ,  0.        ,  0.62      ,
                -0.        ,  0.11061639, -0.        ,  0.        ,  0.00616395,
                1.        ,  0.        ,  0.        ,  1.        ,  0.        ,
                0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                0.        ,  1.        ,  0.        ,  1.        ,  0.        ],
                dtype=float32),
            'player_1': array([
                -1.0110294 , -0.        ,  0.6066176 , -0.        ,  0.7077206 ,
                0.20325357,  0.7077206 , -0.20325357,  0.        , -0.        ,
                0.        , -0.        ,  0.        , -0.        ,  0.        ,
                -0.        ,  1.0110294 ,  0.        ,  0.75827205,  0.        ,
                -0.        ,  0.        , -0.        ,  0.        ,  0.62      ,
                -0.        ,  0.11061639, -0.        ,  0.        ,  0.00616395,
                1.        ,  0.        ,  0.        ,  1.        ,  0.        ,
                0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                0.        ,  1.        ,  0.        ,  0.        ,  1.        ],
            dtype=float32)},
        ...}
        >>> from co_mas.test.utils import vector_sample_sample
        >>> action = {}
        >>> for agent in sync_vec_env.possible_agents:
        ...     agent_envs = sync_vec_env.envs_have_agent(agent)
        ...     if len(agent_envs) > 0:
        ...         action[agent] = vector_sample_sample(
        ...             agent, obs[agent], info[agent], sync_vec_env.action_space(agent), agent_envs
        ...         )
        ...
        >>> pprint(action)
        {'player_0': {'env_0': 6, 'env_1': 14}, 'player_1': {'env_0': 10, 'env_1': 7}}
        >>> pprint(rewards)
        {'player_0': {'env_0': np.float32(0.0), 'env_1': np.float32(0.0)},
        'player_1': {'env_0': np.float32(0.0), 'env_1': np.float32(0.0)}}
        >>> pprint(terminates)
        {'player_0': {'env_0': False, 'env_1': False},
        'player_1': {'env_0': False, 'env_1': False}}
        >>> pprint(truncates)
        {'player_0': {'env_0': False, 'env_1': False},
        'player_1': {'env_0': False, 'env_1': False}}
        >>> pprint(infos)
        {'player_0': {'action_masks': {
                        'env_0': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]),
                        'env_1': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0])},
                    'score_reward': {'env_0': 0, 'env_1': 0}},
         'player_1': {'action_masks': {
                        'env_0': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]),
                        'env_1': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0])},
                    'score_reward': {'env_0': 0, 'env_1': 0}}}
        >>> from co_mas.wrappers.vector import SyncAgentStateVectorParallelEnvWrapper
        >>> as_sync_vec_env= SyncAgentStateVectorParallelEnvWrapper(sync_vec_env)
        >>> pprint(as_sync_vec_env.state())
        {'player_0': {
            'env_0': array([
            -1.0110294 , -0.        ,  0.6066176 , -0.        ,  0.7077206 ,
            0.20325357,  0.7077206 , -0.20325357,  0.        , -0.        ,
            0.        , -0.        ,  0.        , -0.        ,  0.        ,
            -0.        ,  1.0110294 ,  0.        ,  0.75827205,  0.        ,
            -0.        ,  0.        , -0.        ,  0.        ,  0.62      ,
            -0.        ,  0.11061639, -0.        ,  0.        ,  0.00616395,
            1.        ,  0.        ,  0.        ,  1.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  1.        ,  0.        ,  1.        ,  0.        ],
            dtype=float32),
            'env_1': array([
            -1.0110294 , -0.        ,  0.6066176 , -0.        ,  0.7077206 ,
            0.20325357,  0.7077206 , -0.20325357,  0.        , -0.        ,
            0.        , -0.        ,  0.        , -0.        ,  0.        ,
            -0.        ,  1.0110294 ,  0.        ,  0.75827205,  0.        ,
            -0.        ,  0.        , -0.        ,  0.        ,  0.62      ,
            -0.        ,  0.11061639, -0.        ,  0.        ,  0.00616395,
            1.        ,  0.        ,  0.        ,  1.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  1.        ,  0.        ,  1.        ,  0.        ],
            dtype=float32)},
        ...}
        >>> sync_vec_env.close()
    """

    def __init__(self, env_fns: Iterator[Callable[[], ParallelEnv]] | Sequence[Callable[[], ParallelEnv]]):

        self.env_fns = env_fns
        # Initialise all sub-environments
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.env_ids = tuple(f"env_{env_id}" for env_id in range(self.num_envs))
        self._map_env_id_to_env = dict(zip(self.env_ids, self.envs))

        self.metadata = self.envs[0].metadata
        self.single_observation_spaces = self.envs[0].observation_spaces
        self.single_action_spaces = self.envs[0].action_spaces
        self.possible_agents = tuple(self.envs[0].possible_agents)
        self._check_spaces()

        self.observation_spaces = gym.spaces.Dict(
            {
                agent: gym.spaces.Dict({env_id: self.single_observation_spaces[agent] for env_id in self.env_ids})
                for agent in self.possible_agents
            }
        )
        self.action_spaces = gym.spaces.Dict(
            {
                agent: gym.spaces.Dict({env_id: self.single_action_spaces[agent] for env_id in self.env_ids})
                for agent in self.possible_agents
            }
        )
        if hasattr(self.envs[0], "state_space"):
            if not hasattr(self.envs[0], "state_spaces"):
                self.single_state_space = self.envs[0].state_space
                self.state_space = gym.spaces.Dict(
                    {env_id: env.state_space for env_id, env in zip(self.env_ids, self.envs)}
                )

        self.agents = {env_id: tuple(env.agents[:]) for env_id, env in zip(self.env_ids, self.envs)}
        self.agents_old = {env_id: [] for env_id in self.env_ids}
        self._envs_have_agents = defaultdict(list)
        self._update_envs_have_agents()

        self._mark_envs()
        # record which environments will autoreset
        self._autoreset_envs = {env_id: False for env_id in self.env_ids}

    def __repr__(self) -> str:
        """Returns a string representation of the vector environment."""

        return f"SyncVectorParallelEnv(num_envs={self.num_envs})"

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
            if hasattr(self, "single_state_space") and not (env.state_space == self.single_state_space):
                raise RuntimeError(
                    f"Some environments have a state space `{env.state_space}` different from `{self.single_state_space}`. "
                    "In order to batch states, the state spaces from all environments must be equal."
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

        self._need_autoreset_envs = {env_id: _mark_env(env) for env_id, env in zip(self.env_ids, self.envs)}

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

    def reset(
        self, seed: int | list[int] | Dict[EnvID, int] | None = None, options: dict | None = None
    ) -> Tuple[Dict[AgentID, Dict[EnvID, ObsType]], Dict[AgentID, Dict[EnvID, Dict]]]:
        if seed is None:
            seed = {env_id: None for env_id in self.env_ids}
        elif isinstance(seed, int):
            seed = {env_id: seed for env_id in self.env_ids}
        elif isinstance(seed, list):
            seed = {env_id: s for env_id, s in zip(self.env_ids, seed)}

        assert set(seed.keys()) == set(
            self.env_ids
        ), "The key (env_id) of seeds must match the ids of sub-environments."

        observation = {agent: {} for agent in self.possible_agents}
        vector_info = {agent: {} for agent in self.possible_agents}

        for env_id, _seed in seed.items():
            env = self.sub_env(env_id)
            obs, info = env.reset(seed=_seed, options=options)
            for agent in self.possible_agents:
                if agent in obs:
                    observation[agent][env_id] = obs[agent]
            self.add_info_in_place(vector_info, info, env_id)

        self.construct_batch_result_in_place(observation)

        self.agents = {env_id: tuple(env.agents[:]) for env_id, env in zip(self.env_ids, self.envs)}
        self.agents_old = {env_id: [] for env_id in self.env_ids}
        self._envs_have_agents = defaultdict(list)
        self._update_envs_have_agents()

        self._autoreset_envs = {env_id: False for env_id in self.env_ids}

        return observation, vector_info

    def step(self, actions: Dict[AgentID, Dict[EnvID, ActionType]]) -> Tuple[
        Dict[AgentID, Dict[EnvID, ObsType]],
        Dict[AgentID, Dict[EnvID, float]],
        Dict[AgentID, Dict[EnvID, bool]],
        Dict[AgentID, Dict[EnvID, bool]],
        Dict[AgentID, Dict[EnvID, Dict]],
    ]:
        observation = {agent: {} for agent in self.possible_agents}
        reward = {agent: {} for agent in self.possible_agents}
        termination = {agent: {} for agent in self.possible_agents}
        truncation = {agent: {} for agent in self.possible_agents}
        vector_info = {agent: {} for agent in self.possible_agents}

        env_actions = {env_id: {} for env_id in self.env_ids}
        for agent, agent_actions in actions.items():
            for env_id in self._envs_have_agents[agent]:
                env_actions[env_id][agent] = agent_actions[env_id]
        for env_id, env_acts in env_actions.items():
            env = self.sub_env(env_id)
            if self._autoreset_envs[env_id]:
                obs, info = env.reset()
                for agent in env.agents:
                    observation[agent][env_id] = obs[agent]
                    reward[agent][env_id] = 0
                    termination[agent][env_id] = False
                    truncation[agent][env_id] = False
            else:
                env_agents = env.agents
                obs, rew, term, trunc, info = env.step(env_acts)
                for agent in env_agents:
                    observation[agent][env_id] = obs[agent]
                    reward[agent][env_id] = rew[agent]
                    termination[agent][env_id] = term[agent]
                    truncation[agent][env_id] = trunc[agent]
            self.add_info_in_place(vector_info, info, env_id)

        self.construct_batch_result_in_place(observation)
        self.construct_batch_result_in_place(reward)
        self.construct_batch_result_in_place(termination)
        self.construct_batch_result_in_place(truncation)

        self._autoreset_envs = {
            env_id: (len(env.agents) == 0) and self._need_autoreset_envs[env_id]
            for env_id, env in zip(self.env_ids, self.envs)
        }

        self.agents_old = deepcopy(self.agents)
        self.agents = {env_id: tuple(env.agents[:]) for env_id, env in zip(self.env_ids, self.envs)}
        self._update_envs_have_agents()

        return observation, reward, termination, truncation, vector_info

    def close(self):
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True

    def state(self) -> Dict[EnvID, ObsType]:
        if not hasattr(self, "state_space"):
            if hasattr(self.envs[0], "state_spaces"):
                raise RuntimeError(
                    "Please use `AgentStateVectorParallelEnvWrapper` to get the state for each agent since sub-environments have `state_spaces` functions."
                )
            else:
                raise RuntimeError("Sub-environments do not have a `state` function.")
        states = {env_id: env.state() for env_id, env in zip(self.env_ids, self.envs)}
        states = {env_id: env_state for env_id, env_state in states.items() if len(env_state) > 0}
        return states

    @property
    def num_agents(self) -> Dict[EnvID, int]:
        return {env_id: len(env.agents) for env_id, env in zip(self.env_ids, self.envs)}

    def sub_env(self, env_id: EnvID) -> ParallelEnv:
        """
        Return the sub-environment with the given ``env_id``.
        """
        return self._map_env_id_to_env[env_id]

    def __getattr__(self, name: str) -> Tuple:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        if name != "envs" and all(hasattr(env, name) for env in self.envs):
            return tuple(getattr(env, name) for env in self.envs)
        else:
            raise AttributeError(f"attribute '{name}' not found in sub-environments")
