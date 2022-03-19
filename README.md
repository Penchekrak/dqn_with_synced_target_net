basic usage:
`python3 main.py --config-name pong_dqn_cnn_gpu.yaml`

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