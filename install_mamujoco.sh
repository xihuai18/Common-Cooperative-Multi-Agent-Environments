# Multi-agent MuJoCo
pip install gymnasium-robotics mujoco

python -c "from gymnasium_robotics import mamujoco_v1; env = mamujoco_v1.parallel_env('Ant', '4x2'); print(env.reset(seed=42)); print(env.step({agent: env.action_space(agent).sample() for agent in env.agents}))"