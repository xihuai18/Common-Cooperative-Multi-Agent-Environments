Commonly-used Multi-agent Environments Installation, Convenient Wrappers, and VectorEnv Implementation with PettingZoo (and Gymnasium) Compatibility.

## Environment Supports
- [x] [Google Research Football](https://github.com/xihuai18/GFootball-Gymnasium-Pettingzoo)
- [x] [Multi-agent MuJoCo](https://github.com/xihuai18/MaMuJoCo-PettingZoo)
- [x] [StarCraft Multi-Agent Challenge](https://github.com/xihuai18/SMAC-PettingZoo) ([SMAC](https://github.com/oxwhirl/smac) and [SMACv2](https://github.com/oxwhirl/smacv2))

## Installation

The scripts `install_grf.sh`, `install_mamujoco.sh` and `install_smac.sh` include detailed installation guides.

### PyPi from sources

```shell
pip install git+https://github.com/xihuai18/Common-Cooperative-Multi-Agent-Environments.git
```

### Install from GitHub sources
```shell
git clone https://github.com/xihuai18/Common-Cooperative-Multi-Agent-Environments.git
cd Common-Cooperative-Multi-Agent-Environments
pip install -r requirements.txt
pip install .
```

## Parallel Env Wrappers

### State, Observation, Action, Reward Wrappers
- [x] AgentStateParallelEnvWrapper 

### Environment Pipelines
- [x] OrderForcingParallelEnvWrapper
- [x] AutoResetParallelEnvWrapper

### Improved PettingZoo Wrappers
- [x] AECToParallelWrapper
- [x] ParallelToAECWrapper

### Base Environment Abstractions
- [x] ParallelEnv: ParallelEnv with randomness control as in Gymnasium.

## VectorEnv Implementation
- [ ] XXX
