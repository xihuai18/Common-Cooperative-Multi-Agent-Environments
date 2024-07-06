import sys

from gfootball import gfootball_pettingzoo_v1
from loguru import logger
from mamujoco_pettingzoo import mamujoco_pettingzoo_v1
from smac_pettingzoo import smacv1_pettingzoo_v1, smacv2_pettingzoo_v1

from co_mas.test.sync_vector_env import sync_vector_env_test
from co_mas.vector import SyncVectorParallelEnv

logger.remove()
logger.add(sys.stdout, level="INFO")


def env_smacv1_fn():
    return smacv1_pettingzoo_v1.parallel_env("3m")


sync_vec_env = SyncVectorParallelEnv([env_smacv1_fn for _ in range(2)])

sync_vector_env_test(sync_vec_env, num_cycles=1000)


def env_smacv2_fn():
    return smacv2_pettingzoo_v1.parallel_env("10gen_terran_5_vs_5")


sync_vec_env = SyncVectorParallelEnv([env_smacv2_fn for _ in range(2)])

sync_vector_env_test(sync_vec_env, num_cycles=1000)


def env_gfootball_fn():
    return gfootball_pettingzoo_v1.parallel_env("academy_3_vs_1_with_keeper")


sync_vec_env = SyncVectorParallelEnv([env_gfootball_fn for _ in range(2)])

sync_vector_env_test(sync_vec_env, num_cycles=1000)


def env_mamujoco_fn():
    return mamujoco_pettingzoo_v1.parallel_env("Ant", "4x2")


sync_vec_env = SyncVectorParallelEnv([env_mamujoco_fn for _ in range(2)])

sync_vector_env_test(sync_vec_env, num_cycles=1000)
