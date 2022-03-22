import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from callbacks import LogReplayVideoCallback, GC


@hydra.main(config_path='configs')
def main(config: DictConfig):
    model = instantiate(config['model'], _recursive_=False)
    logger = WandbLogger(**config['logger'])
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    callbacks = [GC(), ModelCheckpoint(save_top_k=1, monitor="avg_reward", mode="max", verbose=True)]
    if config['log_video']:
        callbacks.append(LogReplayVideoCallback(config['log_video_path']))
    trainer = Trainer(**config['trainer'], logger=logger, callbacks=callbacks)
    trainer.fit(model)


if __name__ == "__main__":
    main()
