import sys

from gfootball import gfootball_pettingzoo_v1
from loguru import logger
from mamujoco_pettingzoo import mamujoco_pettingzoo_v1
from smac_pettingzoo import smacv1_pettingzoo_v1, smacv2_pettingzoo_v1

from co_mas.test.async_vector_env import async_vector_env_test
from co_mas.vector import AsyncVectorParallelEnv
from co_mas.wrappers import AutoResetParallelEnvWrapper
from co_mas.wrappers.vector import AsyncAgentStateVectorParallelEnvWrapper

logger.remove()
logger.add(sys.stdout, level="INFO")

logger.info("SMACv1")


def env_smacv1_fn():
    return smacv1_pettingzoo_v1.parallel_env("3m")


def env_smacv1_fn_ar():
    return AutoResetParallelEnvWrapper(smacv1_pettingzoo_v1.parallel_env("3m"))


async_vec_env = AsyncVectorParallelEnv([env_smacv1_fn, env_smacv1_fn_ar], debug=True)

async_vec_env = AsyncAgentStateVectorParallelEnvWrapper(async_vec_env)

async_vector_env_test(async_vec_env, num_cycles=500)

logger.info("SMACv2")


def env_smacv2_fn():
    return smacv2_pettingzoo_v1.parallel_env("10gen_terran_5_vs_5")


async_vec_env = AsyncVectorParallelEnv([env_smacv2_fn for _ in range(2)], debug=True)

async_vec_env = AsyncAgentStateVectorParallelEnvWrapper(async_vec_env)

async_vector_env_test(async_vec_env, num_cycles=500)


logger.info("GFootball")


def env_gfootball_fn():
    return gfootball_pettingzoo_v1.parallel_env("academy_3_vs_1_with_keeper", number_of_left_players_agent_controls=2)


async_vec_env = AsyncVectorParallelEnv([env_gfootball_fn for _ in range(2)], debug=True)


async_vec_env = AsyncAgentStateVectorParallelEnvWrapper(async_vec_env)

async_vector_env_test(async_vec_env, num_cycles=500)


logger.info("MAMuJoCo")


def env_mamujoco_fn():
    return mamujoco_pettingzoo_v1.parallel_env("Ant", "4x2")


async_vec_env = AsyncVectorParallelEnv([env_mamujoco_fn for _ in range(2)], debug=True)


async_vec_env = AsyncAgentStateVectorParallelEnvWrapper(async_vec_env)

async_vector_env_test(async_vec_env, num_cycles=500)
