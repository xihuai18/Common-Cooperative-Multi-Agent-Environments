# Multi-agent MuJoCo
pip install git+https://github.com/xihuai18/MaMuJoCo-PettingZoo.git mujoco

python -c "from mamujoco_pettingzoo import mamujoco_pettingzoo_v1; env = mamujoco_pettingzoo_v1.parallel_env('Ant', '4x2'); print(env.reset(seed=42)); print(env.step({agent: env.action_space(agent).sample() for agent in env.agents}))"