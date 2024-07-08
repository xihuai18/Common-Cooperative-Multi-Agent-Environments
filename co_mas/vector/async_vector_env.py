from __future__ import annotations

import multiprocessing
import sys
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Iterator, Sequence, Tuple

import gymnasium as gym
from gymnasium.error import CustomSpaceError
from gymnasium.vector.utils import (
    CloudpickleWrapper,
    clear_mpi_env_vars,
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)
from loguru import logger
from pettingzoo.utils.env import ParallelEnv

from co_mas.vector.vector_env import VectorParallelEnv
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
    """

    def __init__(
        self,
        env_fns: Iterator[Callable[[], ParallelEnv]] | Sequence[Callable[[], ParallelEnv]],
        use_shared_memory: bool = True,
        context: str | None = None,
        daemon: bool = True,
        worker: Callable[[int, Callable[[], ParallelEnv], Connection, Connection, bool, Queue], None] | None = None,
    ):
        """
        Vectorized environment that runs multiple environments in parallel, modified from gymnasium.

        Args:
            env_fns: Functions that create the environments.
            shared_memory: If ``True``, then the observations from the worker processes are communicated back through shared variables. This can improve the efficiency if the observations are large (e.g. images).
            context: Context for `multiprocessing`. If ``None``, then the default context is used.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children, so for some environments you may want to have it set to ``False``.
            worker: If set, then use that worker in a subprocess instead of a default one. Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled.
        """

        self.env_fns = env_fns
        self.use_shared_memory = use_shared_memory

        self.num_envs = len(env_fns)
        self.env_ids = tuple(f"env_{env_id}" for env_id in range(self.num_envs))

        dummy_env = env_fns[0]()
        self.metadata = dummy_env.metadata

        self.single_observation_spaces = dummy_env.observation_spaces
        self.single_action_spaces = dummy_env.action_spaces
        self.possible_agents = dummy_env.possible_agents

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

        if hasattr(dummy_env, "state_space"):
            if not hasattr(dummy_env, "state_spaces"):
                self.single_state_space = dummy_env.state_space
                self.state_space = gym.spaces.Dict(
                    {env_id: deepcopy(self.single_state_space) for env_id in self.env_ids}
                )

        dummy_env.close()
        del dummy_env

        # Generate the multiprocessing context for the observation buffer
        ctx = multiprocessing.get_context(context)
        if self.use_shared_memory:
            try:
                _obs_buffer = create_shared_memory(self.single_observation_space, n=self.num_envs, ctx=ctx)
                self.observations = read_from_shared_memory(self.single_observation_space, _obs_buffer, n=self.num_envs)
                if hasattr(self, "single_state_space"):
                    _state_buffer = create_shared_memory(self.single_state_space, n=self.num_envs, ctx=ctx)
                    self.states = read_from_shared_memory(self.single_observation_space, _obs_buffer, n=self.num_envs)
                else:
                    _state_buffer = None

            except CustomSpaceError as e:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` is incompatible with non-standard Gymnasium observation spaces (i.e. custom spaces inheriting from `gymnasium.Space`), "
                    "and is only compatible with default Gymnasium spaces (e.g. `Box`, `Tuple`, `Dict`) for batching. "
                    "Set `shared_memory=False` if you use custom observation spaces."
                ) from e
        else:
            _obs_buffer = None
            self.observations = {agent: {} for agent in self.possible_agents}
            if hasattr(self, "single_state_space"):
                _state_buffer = None
                self.states = {env_id: {} for env_id in self.env_ids}
            else:
                _state_buffer = None

        self.parent_pipes, self.processes = [], []
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
                        _state_buffer,
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

        # TODO: update envs_have_agents
        # self.agents = {env_id: tuple(env.agents[:]) for env_id, env in zip(self.env_ids, self.envs)}
        self.agents = {}
        for env_id, pipe in self._map_env_id_to_parent_pipe.items():
            pipe.send(("agents", {}))
            agents, _ = pipe.recv()
            self.agents[env_id] = agents
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


def _async_parallel_env_worker(
    index: int,
    env_fn: Callable[[], ParallelEnv],
    pipe: Connection,
    parent_pipe: Connection,
    shared_memory_observation: multiprocessing.Array | Dict[str, Any] | Tuple[Any, ...] | None,
    shared_memory_state: multiprocessing.Array | Dict[str, Any] | Tuple[Any, ...] | None,
    error_queue: Queue,
):
    """
    Worker Process for the `AsyncVectorParallelEnv` that runs the environment and communicates with the main process.
    NOTE: The `autoreset` mechanism is implemented inside the worker process.
    """
    env = env_fn()
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces
    if hasattr(env, "state_spaces"):  # compatible with `agentstate` env
        state_spaces = env.state_spaces
        state_space = None
    elif hasattr(env, "state_space"):
        state_spaces = None
        state_space = env.state_space
    else:
        state_spaces = state_space = None

    need_autoreset = True
    while hasattr(env, "env"):
        if isinstance(env, AutoResetParallelEnvWrapper):
            need_autoreset = False
        env = env.env
    autoreset = False

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
                    reward, terminated, truncated = 0, False, False
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
                    write_to_shared_memory(observation_spaces, index, observation, shared_memory_observation)
                    observation = None

                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "state":
                # TODO: shared_memory for state
                state = env.state()
                if shared_memory_state:
                    if state_spaces:
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
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
