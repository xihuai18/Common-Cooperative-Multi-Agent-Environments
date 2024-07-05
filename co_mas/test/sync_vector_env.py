from pprint import pformat, pprint

from loguru import logger
from tqdm import tqdm

from co_mas.test.utils import vector_sample_sample
from co_mas.vector import SyncVectorParallelEnv


def check_dict(info_dict, num):
    for k, v in info_dict.items():
        if isinstance(v, dict):
            check_dict(v, num)
        else:
            assert len(v) == num, f"Dictionary {k} should have length of 1"


def sync_vector_env_test(sync_vec_env: SyncVectorParallelEnv, num_cycles=1000):
    """
    Test whether the SyncVectorParallelEnv can run an episode.
    """
    sync_vec_env.reset(seed=42)
    # Test contents
    # 1. where the sub-environments reset correctly

    obs, info = sync_vec_env.reset()
    envs_have_agents = sync_vec_env.envs_have_agents

    for _ in tqdm(range(num_cycles)):
        action = {}
        for agent in sync_vec_env.possible_agents:
            agent_envs = envs_have_agents[agent]
            if len(agent_envs) > 0:
                action[agent] = vector_sample_sample(
                    agent, obs[agent], info[agent], sync_vec_env.action_space(agent), agent_envs
                )
        pprint(envs_have_agents)
        pprint(info)
        pprint(action)
        envs_have_agents = sync_vec_env.envs_have_agents

        obs, rew, terminated, truncated, info = sync_vec_env.step(action)
        pprint(sync_vec_env.agents)
        pprint(envs_have_agents)
        pprint(terminated)
        pprint(truncated)

        agents = sync_vec_env.agents
        should_be_reset = [len(_agents) == 0 for _agents in agents]
        print(f"{pformat(should_be_reset)}==?\n{pformat(sync_vec_env._autoreset_envs)}")
        assert all(
            sync_vec_env._autoreset_envs[i] == should_be_reset[i] for i in range(sync_vec_env.num_envs)
        ), "Autoreset environments should be reset"
        for agent in sync_vec_env.possible_agents:
            envs_have_agent = envs_have_agents[agent]
            assert len(obs.get(agent, [])) == len(
                envs_have_agent
            ), "Observations should be the same length as envs_have_agent"
            assert len(rew.get(agent, [])) == len(
                envs_have_agent
            ), "Rewards should be the same length as envs_have_agent"
            assert len(terminated.get(agent, [])) == len(
                envs_have_agent
            ), "Terminations should be the same length as envs_have_agent"
            assert len(truncated.get(agent, [])) == len(
                envs_have_agent
            ), "Truncations should be the same length as envs_have_agent"
            check_dict(info.get(agent, {}), len(envs_have_agent))

    logger.success("SyncVectorParallelEnv Test Passed!")
