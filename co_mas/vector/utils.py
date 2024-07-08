from copy import deepcopy
from typing import Sequence

import gymnasium as gym


def dict_space_from_dict_single_spaces(
    single_spaces: gym.spaces.Dict, agents: Sequence[str], env_ids: Sequence[str]
) -> gym.spaces.Dict:
    """
    Create a Dict space from a dict single spaces.
    """
    return gym.spaces.Dict(
        {agent: gym.spaces.Dict({env_id: deepcopy(single_spaces[agent]) for env_id in env_ids}) for agent in agents}
    )


def dict_space_from_single_space(single_space: gym.Space, env_ids: Sequence[str]) -> gym.spaces.Dict:
    """
    Create a Dict space from a single space.
    """
    return gym.spaces.Dict({env_id: deepcopy(single_space) for env_id in env_ids})
