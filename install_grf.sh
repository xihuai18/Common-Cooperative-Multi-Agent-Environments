# gfootball

## system dependence
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
## pip dependence
pip install -U pip setuptools psutil wheel
conda install anaconda::py-boost -y
## install gfootball
pip install git+https://github.com/xihuai18/GFootball-Gymnasium-Pettingzoo.git

## test
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 
### gymnasium apis
python -c "import gymnasium as gym; import gfootball; env = gym.make('GFootball/academy_3_vs_1_with_keeper-simplev1-v0'); print(env.reset(seed=0)); print(env.step([0]))"

### pettingzoo apis
python -c "from gfootball import gfootball_pettingzoo_v1; env = gfootball_pettingzoo_v1.parallel_env('academy_3_vs_1_with_keeper', representation='simplev1', number_of_left_players_agent_controls=2); print(env.reset(seed=0)); print(env.step({'player_0':0, 'player_1':0}))"
