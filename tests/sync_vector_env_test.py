import sys

from gfootball import gfootball_pettingzoo_v1
from loguru import logger
from mamujoco_pettingzoo import mamujoco_pettingzoo_v1
from smac_pettingzoo import smacv1_pettingzoo_v1, smacv2_pettingzoo_v1

from co_mas.test.sync_vector_env import sync_vector_env_test
from co_mas.vector import SyncVectorParallelEnv
from co_mas.wrappers import AutoResetParallelEnvWrapper
from co_mas.wrappers.vector import SyncAgentStateVectorParallelEnvWrapper

logger.remove()
logger.add(sys.stdout, level="DEBUG")

logger.info("SMACv1")


def env_smacv1_fn():
    return smacv1_pettingzoo_v1.parallel_env("3m")


def env_smacv1_fn_as():
    return AutoResetParallelEnvWrapper(smacv1_pettingzoo_v1.parallel_env("3m"))


sync_vec_env = SyncVectorParallelEnv([env_smacv1_fn, env_smacv1_fn_as], debug=True)

sync_vec_env = SyncAgentStateVectorParallelEnvWrapper(sync_vec_env)

sync_vector_env_test(sync_vec_env, num_cycles=500)


logger.info("SMACv2")


def env_smacv2_fn():
    return smacv2_pettingzoo_v1.parallel_env("10gen_terran_5_vs_5")


sync_vec_env = SyncVectorParallelEnv([env_smacv2_fn for _ in range(2)], debug=True)

sync_vec_env = SyncAgentStateVectorParallelEnvWrapper(sync_vec_env)

sync_vector_env_test(sync_vec_env, num_cycles=500)


logger.info("GFootball")


def env_gfootball_fn():
    return gfootball_pettingzoo_v1.parallel_env("academy_3_vs_1_with_keeper", number_of_left_players_agent_controls=2)


sync_vec_env = SyncVectorParallelEnv([env_gfootball_fn for _ in range(2)], debug=True)


sync_vec_env = SyncAgentStateVectorParallelEnvWrapper(sync_vec_env)

sync_vector_env_test(sync_vec_env, num_cycles=500)


logger.info("MAMuJoCo")


def env_mamujoco_fn():
    return mamujoco_pettingzoo_v1.parallel_env("Ant", "4x2")


sync_vec_env = SyncVectorParallelEnv([env_mamujoco_fn for _ in range(2)], debug=True)


sync_vec_env = SyncAgentStateVectorParallelEnvWrapper(sync_vec_env)

sync_vector_env_test(sync_vec_env, num_cycles=500)
