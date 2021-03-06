import gc
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from gym.wrappers import RecordVideo
from pl_bolts.models.rl import DQN
from pytorch_lightning import Callback


class LogReplayVideoCallback(Callback):
    def __init__(self, path: str = 'video.mp4'):
        self.path = path

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: DQN, **kwargs) -> None:
        with RecordVideo(pl_module.test_env, video_folder=self.path) as recorded_env:
            _ = pl_module.run_n_episodes(recorded_env, 1, 0.0)
        video_paths = [s for s in Path(self.path).iterdir() if s.suffix == '.mp4']
        for path in video_paths:
            trainer.logger.experiment.log({'replay': wandb.Video(str(path))})


class GC(Callback):
    def __init__(self, clean_every_n_epochs: int = 1):
        self.clean_every_n_epochs = clean_every_n_epochs

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: DQN, **kwargs) -> None:
        if trainer.current_epoch + 1 % self.clean_every_n_epochs == 0:
            gc.collect()
            torch.cuda.empty_cache()
