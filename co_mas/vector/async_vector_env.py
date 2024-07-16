from __future__ import annotations

import multiprocessing
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import partial
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from gymnasium.error import AlreadyPendingCallError, CustomSpaceError, NoAsyncCallError
from gymnasium.vector.utils import (
    CloudpickleWrapper,
    clear_mpi_env_vars,
    create_empty_array,
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)
from loguru import logger
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

from co_mas.vector.utils import (
    dict_space_from_dict_single_spaces,
    dict_space_from_single_space,
)
from co_mas.vector.vector_env import EnvID, VectorParallelEnv
from co_mas.wrappers import AutoResetParallelEnvWrapper


class AsyncState(Enum):
    """The AsyncVectorEnv possible states given the different actions."""

    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorParallelEnv(VectorParallelEnv):
    """
    Vectorized PettingZoo Parallel environment that runs multiple environments in parallel.

    It uses `multiprocessing` processes, and pipes for communication, and uses shared memory for acceleration by default.

    Example:
        >>> from co_mas.vector import AsyncVectorParallelEnv
        >>> from smac_pettingzoo import smacv1_pettingzoo_v1
        >>> def env_smacv1_fn():
        ...     return smacv1_pettingzoo_v1.parallel_env("3m")
        ...
        >>> async_vec_env = AsyncVectorParallelEnv([env_smacv1_fn, env_smacv1_fn], debug=True)

        >>> async_vec_env
        AsyncVectorParallelEnv(num_envs=2)
        >>> obs, info = async_vec_env.reset(seed=42)
        >>> from pprint import pprint
        >>> pprint(obs)
        {'marine_0': {'env_0': array([1.        , 0.0764974 , 0.        , 0.0764974 , 1.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 1.        ,
        0.10818365, 0.0764974 , 0.0764974 , 1.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 0.        , 0.        ,
        0.        , 1.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 1.        , 0.        , 0.        ], dtype=float32),
                'env_1': array([1.        , 0.0764974 , 0.        , 0.0764974 , 1.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 1.        ,
        0.10818365, 0.0764974 , 0.0764974 , 1.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 0.        , 0.        ,
        0.        , 1.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 1.        , 0.        , 0.        ], dtype=float32)},
        ...}
        >>> pprint(info)
        {'marine_0': {'action_mask': {
                        'env_0': array([0, 1, 1, 1, 1, 1, 0, 0, 0]),
                        'env_1': array([0, 1, 1, 1, 1, 1, 0, 0, 0])}},
        ...}
        >>> state = async_vec_env.state()
        RuntimeError: Please use `AgentStateVectorParallelEnvWrapper` to get the state for each agent since sub-environments have `state_spaces` functions.
        >>> from co_mas.test.utils import vector_sample_sample
        >>> action = {}
        >>> for agent in async_vec_env.possible_agents:
        ...      agent_envs = async_vec_env.envs_have_agent(agent)
        ...      if len(agent_envs) > 0:
        ...          action[agent] = vector_sample_sample(
        ...              agent, obs[agent], info[agent], async_vec_env.action_space(agent), agent_envs
        ...          )
        ...
        >>> pprint(action)
        {'marine_0': {'env_0': np.int64(5), 'env_1': np.int64(1)},
        'marine_1': {'env_0': np.int64(4), 'env_1': np.int64(4)},
        'marine_2': {'env_0': np.int64(4), 'env_1': np.int64(2)}}
        >>> observations, rewards, terminates, truncates, infos = async_vec_env.step(action)
    """

    def __init__(
        self,
        env_fns: Iterator[Callable[[], ParallelEnv]] | Sequence[Callable[[], ParallelEnv]],
        use_shared_memory: bool = True,
        context: str | None = None,
        daemon: bool = True,
        worker: Callable[[int, Callable[[], ParallelEnv], Connection, Connection, bool, Queue], None] | None = None,
        debug: bool = False,
    ):
        """
        Vectorized environment that runs multiple environments in parallel, modified from gymnasium.

        Args:
            env_fns: Functions that create the environments.
            shared_memory: If ``True``, then the observations from the worker processes are communicated back through shared variables. This can improve the efficiency if the observations are large (e.g. images).
            context: Context for `multiprocessing`. If ``None``, then the default context is used.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children, so for some environments you may want to have it set to ``False``.
            worker: If set, then use that worker in a subprocess instead of a default one. Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled.
            debug: Whether to add assertions, which will slow down the environment.
        """

        self.env_fns = env_fns
        self.use_shared_memory = use_shared_memory
        self.debug = debug

        self.num_envs = len(env_fns)
        self.env_ids = tuple(f"env_{env_id}" for env_id in range(self.num_envs))

        self.dummy_env = env_fns[0]()
        self.metadata = self.dummy_env.metadata

        self.single_observation_spaces = self.dummy_env.observation_spaces
        self.single_action_spaces = self.dummy_env.action_spaces
        self.possible_agents = self.dummy_env.possible_agents

        self.observation_spaces = dict_space_from_dict_single_spaces(
            self.single_observation_spaces, self.possible_agents, self.env_ids
        )
        self.action_spaces = dict_space_from_dict_single_spaces(
            self.single_action_spaces, self.possible_agents, self.env_ids
        )

        self.single_state_space = None
        self.state_space = None
        self._single_state_spaces = None
        self._state_spaces = None
        if hasattr(self.dummy_env, "state_space"):
            if not hasattr(self.dummy_env, "state_spaces"):
                self.single_state_space = self.dummy_env.state_space
                self.state_space = dict_space_from_single_space(self.single_state_space, self.env_ids)
            else:
                # compatible with AgentStateParallelEnvWrapper
                self._single_state_spaces = self.dummy_env.state_spaces
                self._state_spaces = dict_space_from_dict_single_spaces(
                    self._single_state_spaces, self.possible_agents, self.env_ids
                )

        self.dummy_env.close()

        # Generate the multiprocessing context for the observation buffer
        ctx = multiprocessing.get_context(context)
        if self.use_shared_memory:
            try:
                _obs_buffer = create_shared_memory(self.single_observation_spaces, n=self.num_envs, ctx=ctx)
                """ 
                {
                    AgentID: multiprocessing.Array of shape np.prod(`num_envs`, *`single_observation_space`)
                } 
                """
                self.observations: tuple[dict[AgentID, Any]] = read_from_shared_memory(
                    self.single_observation_spaces, _obs_buffer, n=self.num_envs
                )
                """ 
                (
                    AgentID 0: np.array of shape `single_observation_space`,
                    ...,
                    AgentID n: np.array of shape `single_observation_space`
                )
                """
                _state_buffer = None
                self.states = None
                _agent_state_buffer = None
                self._agent_states = None
                if self.single_state_space is not None:
                    _state_buffer = create_shared_memory(self.single_state_space, n=self.num_envs, ctx=ctx)
                    """ 
                    {
                        AgentID: multiprocessing.Array of shape np.prod(`num_envs`, *`single_state_space`)
                    } 
                    """
                    self.states = read_from_shared_memory(self.single_state_space, _state_buffer, n=self.num_envs)
                    """ 
                    (
                        AgentID 0: np.array of shape `single_state_space`,
                        ...,
                        AgentID n: np.array of shape `single_state_space`
                    )
                    """
                elif self._single_state_spaces is not None:
                    _agent_state_buffer = create_shared_memory(self._single_state_spaces, n=self.num_envs, ctx=ctx)
                    self._agent_states = read_from_shared_memory(
                        self._single_state_spaces, _agent_state_buffer, n=self.num_envs
                    )

            except CustomSpaceError as e:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` is incompatible with non-standard Gymnasium observation spaces (i.e. custom spaces inheriting from `gymnasium.Space`), "
                    "and is only compatible with default Gymnasium spaces (e.g. `Box`, `Tuple`, `Dict`) for batching. "
                    "Set `shared_memory=False` if you use custom observation spaces."
                ) from e
        else:
            _obs_buffer = None
            self.observations: tuple[dict[AgentID, Any]] = None
            _state_buffer = None
            self.states = None
            _agent_state_buffer = None
            self._agent_states = None

        self.parent_pipes: list[Connection] = []
        self.processes: list[multiprocessing.Process] = []
        self.error_queue = ctx.Queue()
        target = worker or _async_parallel_env_worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        _state_buffer or _agent_state_buffer,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()

        self._map_env_id_to_parent_pipe = dict(zip(self.env_ids, self.parent_pipes))
        self._map_env_id_to_process = dict(zip(self.env_ids, self.processes))

        self._update_agents()
        self.agents_old = {env_id: [] for env_id in self.env_ids}
        self._envs_have_agents = defaultdict(list)
        self._update_envs_have_agents()

    def __repr__(self) -> str:
        """Returns a string representation of the vector environment."""

        return f"AsyncVectorParallelEnv(num_envs={self.num_envs})"

    def _check_spaces(self):
        if hasattr(self, "single_state_space"):
            spaces = (self.single_observation_spaces, self.single_action_spaces, self.single_state_space)
        else:
            spaces = (self.single_observation_spaces, self.single_action_spaces)

        for pipe in self.parent_pipes:
            pipe.send(("_check_spaces", spaces))

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        if len(spaces) == 3:
            same_observation_spaces, same_action_spaces, same_state_spaces = zip(*results)
        else:
            same_observation_spaces, same_action_spaces = zip(*results)
            same_state_spaces = None

        if not all(same_observation_spaces):
            raise RuntimeError(
                f"Some environments have an observation space different from `{self.single_observation_space}`. "
                "In order to batch observations, the observation spaces from all environments must be equal."
            )
        if not all(same_action_spaces):
            raise RuntimeError(
                f"Some environments have an action space different from `{self.single_action_space}`. "
                "In order to batch actions, the action spaces from all environments must be equal."
            )
        if same_state_spaces is not None and not all(same_state_spaces):
            raise RuntimeError(
                f"Some environments have a state space different from `{self.single_state_space}`. "
                "In order to batch states, the state spaces from all environments must be equal."
            )

    def _raise_if_errors(self, successes: list[bool] | tuple[bool]):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()

            logger.error(f"Received the following error from Worker-{index}: {exctype.__name__}: {value}")
            logger.error(f"Shutting down Worker-{index}.")

            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                logger.error("Raising the last exception back to the main process.")
                raise exctype(value)

    def _update_agents(self) -> None:
        for env_id, pipe in self._map_env_id_to_parent_pipe.items():
            pipe.send(("agents", {}))
        self.agents = {}
        successes = []
        for env_id, pipe in self._map_env_id_to_parent_pipe.items():
            agents, success = pipe.recv()
            successes.append(success)
            if success:
                self.agents[env_id] = agents
        self._raise_if_errors(successes)

    def reset(
        self, seed: int | list[int] | dict[Any, int] | None = None, options: dict | None = None
    ) -> tuple[dict[Any, dict] | dict[Any, dict[Any, dict]]]:
        if seed is None:
            seed = {env_id: None for env_id in self.env_ids}
        elif isinstance(seed, int):
            seed = {env_id: seed for env_id in self.env_ids}
        elif isinstance(seed, list):
            seed = {env_id: s for env_id, s in zip(self.env_ids, seed)}

        assert set(seed.keys()) == set(
            self.env_ids
        ), "The key (env_id) of seeds must match the ids of sub-environments."
        self._reset_async(seed, options)
        return self._reset_await()

    def _reset_async(self, seed: dict[Any, int], options: dict | None = None) -> None:
        """
        Send calls to the `reset` methods of the sub-environments.

        To get the results of these calls, you may invoke `_reset_wait`.
        """
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                str(self._state.value),
            )
        for env_id, _seed in seed.items():
            pipe = self._map_env_id_to_parent_pipe[env_id]
            pipe.send(("reset", {"seed": _seed, "options": options}))
        self._state = AsyncState.WAITING_RESET

    def _reset_await(self, timeout: float | None = None) -> tuple[dict[Any, dict] | dict[Any, dict[Any, dict]]]:
        """
        Await the results of the `reset` calls sent by `_reset_async`.
        """
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )
        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(f"The call to `reset_wait` has timed out after {timeout} second(s).")

        observation = {agent: {} for agent in self.possible_agents}
        vector_info = {agent: {} for agent in self.possible_agents}

        successes = []
        obses = []
        for i, env_id in enumerate(self.env_ids):
            pipe = self._map_env_id_to_parent_pipe[env_id]
            (obs, info), success = pipe.recv()
            successes.append(success)
            if success:
                if self.debug:
                    if not self.use_shared_memory:
                        self._check_containing_agents(self.agents_old[env_id], obs)
                obses.append(obs)
                self.add_info_in_place(vector_info, info, env_id)
        self._raise_if_errors(successes)

        self._update_agents()
        self.agents_old = {env_id: [] for env_id in self.env_ids}
        self._envs_have_agents = defaultdict(list)
        self._update_envs_have_agents()

        for env_id, obs in zip(self.env_ids, obses):
            for agent in self.agents[env_id]:
                if self.use_shared_memory:
                    observation[agent][env_id] = self.observations[i][agent]
                else:
                    observation[agent][env_id] = obs[agent]
        self.construct_batch_result_in_place(observation)

        self.agents_old = deepcopy(self.agents)

        self._state = AsyncState.DEFAULT
        return observation, vector_info

    def state(self, timeout: float | None = None) -> dict[EnvID, ObsType]:
        """
        Return the state of all sub environments.
        """
        if self.state_space is None:
            if hasattr(self.dummy_env, "state_spaces"):
                raise RuntimeError(
                    "Please use `AgentStateVectorParallelEnvWrapper` to get the state for each agent since sub-environments have `state_spaces` functions."
                )
            else:
                raise NotImplementedError("Sub-environments do not have a `state` function.")
        assert not self.use_shared_memory or self.states is not None, "Shared memory is not enabled."

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `state` while waiting for a pending call to `{self._state.value}` to complete",
                str(self._state.value),
            )
        for pipe in self.parent_pipes:
            pipe.send(("state", {}))

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(f"The call to `call_wait` has timed out after {timeout} second(s).")

        successes = []
        state = {}
        for i, env_id in enumerate(self.env_ids):
            pipe = self._map_env_id_to_parent_pipe[env_id]
            _state, success = pipe.recv()
            successes.append(success)
            if success:
                if self.use_shared_memory and self.states is not None:
                    state[env_id] = self.states[i]
                else:
                    state[env_id] = _state

        self._raise_if_errors(successes)

        self._state = AsyncState.DEFAULT
        return state

    def _agent_state(self) -> dict[AgentID, dict[EnvID, ObsType]]:
        """
        Return the state of all sub environments.
        """
        assert (
            hasattr(self.dummy_env, "state_spaces")
            and hasattr(self.dummy_env, "state_space")
            and hasattr(self.dummy_env, "state")
        ), "Sub-environments should have `state`, `state_spaces` and `state_space` attributes."

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `state` while waiting for a pending call to `{self._state.value}` to complete",
                str(self._state.value),
            )
        for pipe in self.parent_pipes:
            pipe.send(("state", {}))

        assert not self.use_shared_memory or self._agent_states is not None, "Shared memory is not enabled."

        successes = []
        state = {agent: {} for agent in self.possible_agents}
        for i, env_id in enumerate(self.env_ids):
            pipe = self._map_env_id_to_parent_pipe[env_id]
            _state, success = pipe.recv()
            successes.append(success)
            if success:
                if self.debug:
                    if not self.use_shared_memory:
                        self._check_containing_agents(self.agents_old[env_id], _state)
                for agent in self.agents_old[env_id]:
                    if self.use_shared_memory and self._agent_states is not None:
                        state[agent][env_id] = self._agent_states[i][agent]
                    else:
                        state[agent][env_id] = _state[agent]

        self._raise_if_errors(successes)

        self._state = AsyncState.DEFAULT
        return state

    def step(self, actions: dict[AgentID, dict[EnvID, ActionType]]) -> tuple[
        dict[AgentID, dict[EnvID, ObsType]],
        dict[AgentID, dict[EnvID, float]],
        dict[AgentID, dict[EnvID, bool]],
        dict[AgentID, dict[EnvID, bool]],
        dict[AgentID, dict[EnvID, dict]],
    ]:
        self._step_async(actions)
        return self._step_await()

    def _step_async(self, actions: dict[AgentID, dict[EnvID, ActionType]]) -> None:
        """
        Send calls to the `step` methods of the sub-environments.

        To get the results of these calls, you may invoke `_step_wait`.
        """
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete",
                str(self._state.value),
            )
        env_actions = {env_id: {} for env_id in self.env_ids}
        for agent, agent_actions in actions.items():
            for env_id in self._envs_have_agents[agent]:
                env_actions[env_id][agent] = agent_actions[env_id]
        for env_id, action in env_actions.items():
            pipe = self._map_env_id_to_parent_pipe[env_id]
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def _step_await(self, timeout: float | None = None) -> tuple[
        dict[AgentID, dict[EnvID, ObsType]],
        dict[AgentID, dict[EnvID, float]],
        dict[AgentID, dict[EnvID, bool]],
        dict[AgentID, dict[EnvID, bool]],
        dict[AgentID, dict[EnvID, dict]],
    ]:
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(f"The call to `step_wait` has timed out after {timeout} second(s).")

        observation = {agent: {} for agent in self.possible_agents}
        reward = {agent: {} for agent in self.possible_agents}
        termination = {agent: {} for agent in self.possible_agents}
        truncation = {agent: {} for agent in self.possible_agents}
        vector_info = {agent: {} for agent in self.possible_agents}

        successes = []
        reset_agents: dict[EnvID, bool] = {env_id: False for env_id in self.env_ids}
        for i, env_id in enumerate(self.env_ids):
            pipe = self._map_env_id_to_parent_pipe[env_id]
            (obs, rew, term, trunc, info), success = pipe.recv()
            successes.append(success)
            if success:
                if len(self.agents[env_id]) == 0:  # will be reset
                    reset_agents[env_id] = True
                    env_agents = self.possible_agents
                else:
                    env_agents = self.agents[env_id]
                if self.debug:
                    if not self.use_shared_memory:
                        self._check_containing_agents(env_agents, obs)
                    self._check_containing_agents(env_agents, rew)
                    self._check_containing_agents(env_agents, term)
                    self._check_containing_agents(env_agents, trunc)
                    self._check_containing_agents(env_agents, info)
                for agent in env_agents:
                    if self.use_shared_memory:
                        observation[agent][env_id] = self.observations[i][agent]
                    else:
                        observation[agent][env_id] = obs[agent]
                    reward[agent][env_id] = rew[agent]
                    termination[agent][env_id] = term[agent]
                    truncation[agent][env_id] = trunc[agent]
                self.add_info_in_place(vector_info, info, env_id)

        self.construct_batch_result_in_place(observation)
        self.construct_batch_result_in_place(reward)
        self.construct_batch_result_in_place(termination)
        self.construct_batch_result_in_place(truncation)

        self.agents_old = deepcopy(self.agents)
        self._update_agents()
        self._update_envs_have_agents()

        for env_id, _reset in reset_agents.items():
            if _reset:
                self.agents_old[env_id] = deepcopy(self.agents[env_id])

        self._state = AsyncState.DEFAULT

        return observation, reward, termination, truncation, vector_info

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Call a method from each sub environment with args and kwargs.

        Args:
            name (str): Name of the method or property to call.
            *args: Position arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each sub-environment.
        """
        self._call_async(name, *args, **kwargs)
        return self._call_await()

    def _call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.
        """
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `call_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def _call_await(self, timeout: int | float | None = None) -> dict[EnvID, Any]:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out.
                If ``None`` (default), the call to :meth:`step_wait` never times out.
        """

        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(f"The call to `call_wait` has timed out after {timeout} second(s).")

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return dict(zip(self.env_ids, results))

    def _poll_pipe_envs(self, timeout: int | None = None):
        if timeout is None:
            return True

        end_time = time.perf_counter() + timeout
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)

            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    @property
    def num_agents(self) -> dict[EnvID, int]:
        return {env_id: len(self.agents[env_id]) for env_id in self.env_ids}

    def close_extras(self, timeout: int | float | None = None, terminate: bool = False) -> None:
        """
        Close the environments & clean up the extra resources (processes and pipes).
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warning(
                    f"Calling `close` while waiting for a pending call to `{self._state.value}` to complete."
                )
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except multiprocessing.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()
        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)


def _async_parallel_env_worker(
    index: int,
    env_fn: Callable[[], ParallelEnv],
    pipe: Connection,
    parent_pipe: Connection,
    shared_memory_observation: multiprocessing.Array | dict[str, Any] | tuple[Any, ...] | None,
    shared_memory_state: multiprocessing.Array | dict[str, Any] | tuple[Any, ...] | None,
    error_queue: Queue,
):
    """
    Worker Process for the `AsyncVectorParallelEnv` that runs the environment and communicates with the main process.
    NOTE: The `autoreset` mechanism is implemented inside the worker process.
    """
    env = env_fn()
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    state_spaces = getattr(env, "state_spaces", None)
    state_space = getattr(env, "state_space", None)

    if state_spaces is not None:  # compatible with `agentstate` env
        state_space = None

    need_autoreset = True
    _env = env
    while hasattr(env, "env"):
        if isinstance(env, AutoResetParallelEnvWrapper):
            need_autoreset = False
        env = env.env
    env = _env
    autoreset = False

    logger.debug(f"Async Worker Index {index}: {need_autoreset=}")

    parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()

            if command == "reset":
                observation, info = env.reset(**data)
                if shared_memory_observation:
                    write_to_shared_memory(observation_spaces, index, observation, shared_memory_observation)
                    observation = None
                    autoreset = False
                pipe.send(((observation, info), True))
            elif command == "step":
                if autoreset:
                    observation, info = env.reset()
                    reward, terminated, truncated = {}, {}, {}
                    for agent in env.agents:
                        reward[agent], terminated[agent], truncated[agent] = 0.0, False, False
                else:
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(data)
                autoreset = (len(env.agents) == 0) and need_autoreset

                if shared_memory_observation:
                    for agent in env.possible_agents:
                        if agent not in observation:
                            observation[agent] = create_empty_array(
                                observation_spaces[agent], fn=partial(np.full, fill_value=np.nan)
                            )
                    write_to_shared_memory(observation_spaces, index, observation, shared_memory_observation)
                    observation = None

                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "state":
                state = env.state()
                if shared_memory_state:
                    if state_spaces:
                        for agent in env.possible_agents:
                            if agent not in state:
                                state[agent] = create_empty_array(
                                    state_spaces[agent], fn=partial(np.full, fill_value=np.nan)
                                )
                        write_to_shared_memory(state_spaces, index, state, shared_memory_state)
                    elif state_space:
                        write_to_shared_memory(state_space, index, state, shared_memory_state)
                    else:
                        raise ValueError("`state_space` and `state_spaces` are not defined.")
                    state = None
                pipe.send((state, True))
            elif command == "agents":
                pipe.send((env.agents, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "state", "agents", "close", "_setattr", "_check_spaces"]:
                    raise ValueError(f"Trying to call function `{name}` with `call`, use `{name}` directly instead.")

                attr = getattr(env, name)
                if callable(attr):
                    pipe.send((attr(*args, **kwargs), True))
                else:
                    pipe.send((attr, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":

                if len(data) == 2:
                    pipe.send(
                        (
                            (data[0] == observation_spaces, data[1] == action_spaces),
                            True,
                        )
                    )
                else:
                    pipe.send(
                        (
                            (
                                data[0] == observation_spaces,
                                data[1] == action_spaces,
                                data[2] == state_space or data[2] == state_spaces,
                            ),
                            True,
                        )
                    )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must be one of [`reset`, `step`, `state`, `agents`, `close`, `_call`, `_setattr`, `_check_spaces`]."
                )
    except (KeyboardInterrupt, Exception) as e:
        exc_message = traceback.format_exc()
        error_queue.put((index, type(e), exc_message))
        pipe.send((None, False))
    finally:
        env.close()
