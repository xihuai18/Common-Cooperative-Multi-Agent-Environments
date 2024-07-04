from smac_pettingzoo import smacv1_pettingzoo_v1

from co_mas.vector import SyncVectorParallelEnv


def env_fn():
    return smacv1_pettingzoo_v1.parallel_env("3m")


sync_vec_env = SyncVectorParallelEnv([env_fn for _ in range(3)])
