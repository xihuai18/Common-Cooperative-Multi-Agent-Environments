from pprint import pformat
from typing import Any, Dict

from loguru import logger


def _merge(_vector_info: Dict, _env_info: Dict, env_id: Any):
    """
    Merge values in _env_info into _vector_info.
    """
    for k, v in _env_info.items():
        if isinstance(v, dict):
            _merge(_vector_info[k], v, env_id)
        else:
            if k not in _vector_info:
                _vector_info[k] = {env_id: v}
            else:
                _vector_info[k][env_id] = v


infos = (
    {"a": {"k1": 1, "k2": 2}, "b": {"k1": 1, "k2": 3}},
    {"a": {"k1": 1, "k2": 2}, "c": {"k1": 1, "k2": 3}},
    {"b": {"k1": 1, "k2": 2}, "c": {"k1": 1, "k2": 3}},
)

vector_info = {}
possible_keys = ("a", "b", "c")

for info_i, info in enumerate(infos):
    for key in possible_keys:
        if key in info:
            _merge(vector_info, info[key], f"info_{info_i}")

    logger.info("\n" + pformat(vector_info))
