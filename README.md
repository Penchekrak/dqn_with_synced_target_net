Machine Leraning course project 2022

Project: "Research on the usage of target networks in deep reinforcement learning"

Team: Andrey Spiridonov, Vladislav Trifonov, Yerassyl Balkybek, Nikolay Sheyko, Dmitrii Gromyko


This project is dedicated to studying the popular model-free reinforcement learning algorithm - Q-learning. Models under investigation such as DQN, Double DQN, and Dueling DQN has an auxiliary network within its architecture to increase stability of training - target network. We conducted several experiments, which shows that replacing target network with action network leads to instability of DQN and Double DQN training. Nevertheless, it can be removed in Dueling DQN model without loss in stability and achieved score.

The main contributions of this work are as follows:
1. For DQN and Double DQN models the necessity of target network was proven.
2. Direct correlation between the neural network size and stability for Q-learning models without target network was observed.
3. It was shown, that Dueling DQN model without target network can achieve the same performance as original Dueling DQN.

basic usage:
`python3 main.py --config-name pong_dqn_cnn_gpu.yaml`

jupyter notebook with and example of run: `example_of_run.ipynb` and 'dqn.ipynb'

config structure:

```yaml
model:  # here you describe env, agent and underlying net
  _target_: pl_bolts.models.rl.DQN  # standard dqn from pl_bolts
  env: "Pong-v0"  # gym env identifier
  network:  # underlying net with all it's parameters below
    _target_: pl_bolts.models.rl.common.networks.CNN
trainer:  # trainer args from pytorch_lightning
  accelerator: "cpu"  
# following leave as is for simplicity :)
log_video: True
log_video_path: "video"
logger:
  project: "project"
  entity: "skoltech_ml2022_project_synced_target_nets"
```
possible `_target_`s for model and network are class names (with fully resolved module path)
that can be described in `models.py` or `networks.py` or elsewhere. Network class constructor must have first two positional arguments `input_shape: Tuple[int]` and `n_actions: int
`
