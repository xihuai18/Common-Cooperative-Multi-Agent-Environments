from smac_pettingzoo import smacv1_pettingzoo_v1

from co_mas.test.sync_vector_env import sync_vector_env_test
from co_mas.vector import SyncVectorParallelEnv


def env_fn():
    return smacv1_pettingzoo_v1.parallel_env("3m")


sync_vec_env = SyncVectorParallelEnv([env_fn for _ in range(2)])

# obs, info = sync_vec_env.reset(42)
# action = {}

# for agent in sync_vec_env.possible_agents:
#     agent_envs = sync_vec_env.envs_have_agent(agent)
#     if len(agent_envs) > 0:
#         action[agent] = vector_sample_sample(
#             agent, obs[agent], info[agent], sync_vec_env.action_space(agent), agent_envs
#         )

# sync_vec_env.step(action)

sync_vector_env_test(sync_vec_env, num_cycles=1000)
