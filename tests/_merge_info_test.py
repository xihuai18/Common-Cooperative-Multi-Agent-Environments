from pprint import pformat
from typing import Any, Dict, Tuple

from loguru import logger


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


infos = (
    {"a": {"k1": 1, "k2": 2}, "b": {"k1": 1, "k2": 3}},
    {"a": {"k1": 1, "k2": 2}, "c": {"k1": 1, "k2": 3}},
    {"b": {"k1": 1, "k2": 2}, "c": {"k1": 1, "k2": 3}},
)

value_info = {}
possible_keys = ("a", "b", "c")
for key in possible_keys:
    infos_for_key = tuple(info[key] for info in infos if key in info)
    logger.info(pformat(infos_for_key))
    if len(infos_for_key) <= 0:
        continue
    value_info[key] = {}
    dummy_info = infos_for_key[0]
    for _k in dummy_info.keys():
        value_info[key][_k] = _merge(_infos=infos_for_key, _key=_k)

logger.info("\n" + pformat(value_info))
