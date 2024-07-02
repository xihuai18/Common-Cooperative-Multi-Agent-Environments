from typing import Any

import numpy as np
from gymnasium.utils import seeding
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from pettingzoo.utils.env import ParallelEnv as _ParallelEnv

__all__ = ["ParallelEnv"]


class ParallelEnv(_ParallelEnv[AgentID, ObsType, ActionType]):
    """
    We implement this base ParallelEnv class to include some fundamental components in gymnasium.
    NOTE: You can use this `ParallelEnv` instead of the pettingzoo version if the underlying environment of your custom multi-agent environment is not a gymnasium environment.
    """

    # Seeding for reproducibility
    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None  # will be `-1` if the seed is unknown for a given rng

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

    @property
    def np_random_seed(self) -> int:
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random_seed

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, rng: np.random.Generator) -> None:
        self._np_random = rng
        self._np_random_seed = -1

    def state(self) -> np.ndarray:
        """Returns the state.

        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        assert hasattr(self, "state_space"), "Environment does not have a state_space attribute"
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def __del__(self):
        """Forcing the environment to close."""
        self.close()

    def __exit__(self, *args: Any) -> bool:
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False
