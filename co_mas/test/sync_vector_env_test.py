from co_mas.vector import SyncVectorParallelEnv


def sync_vector_env_test(sync_vec_env: SyncVectorParallelEnv, num_cycles=1000):
    """
    Test whether the SyncVectorParallelEnv can run an episode.
    """
    sync_vec_env.reset(seed=42)
    # Test contents
    # 1. where the sub-environments reset correctly

    MAX_RESETS = 2
    for _ in range(MAX_RESETS):
        obs, info = sync_vec_env.reset()

        [False for _ in range(sync_vec_env.num_envs)]

        for _ in range(num_cycles):
            pass
