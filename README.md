This project is dedicated to studying the popular model-free reinforcement learning algorithm - Q-learning. Models under investigation such as DQN, Double DQN, and Dueling DQN has an auxiliary network within its architecture to increase stability of training - target network. We conducted several experiments, which shows that replacing target network with action network leads to instability of DQN and Double DQN training. Nevertheless, it can be removed in Dueling DQN model without loss in stability and achieved score.

For the detailed information check the presentation "ATARI (Do,DU)DQN" and "ML Course project 2022.pdf" report

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
Conclusions:
Results of replacing target network with action network in the training process of Q-learning algorithms vary with respect to setup. For each used environment it's proven to be a bad idea for DQN model. The case of Double DQN is not so simple and requires further investigation. For Dueling DQN model was shown, that modified algorithm can perform on the same level as its basemodel. More prolonged training justify the results for Dueling DQN - model without training network perform on the same level as PyTorch basemodel. Also, correlation of depth of neural network and stability of target-network-less model was revealed - large network decrease variance of loss function during training.
