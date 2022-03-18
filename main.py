import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from callbacks import LogReplayVideoCallback


@hydra.main(config_path='configs')
def main(config: DictConfig):
    model = instantiate(config['model'], _recursive_=False)
    logger = WandbLogger(**config['logger'])
    if config['log_video']:
        callbacks = [LogReplayVideoCallback(config['log_video_path'])]
    else:
        callbacks = []
    trainer = Trainer(**config['trainer'], logger=logger, callbacks=callbacks)
    trainer.fit(model)


if __name__ == "__main__":
    main()
