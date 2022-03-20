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

    for key_1 in config['model'].keys():
        if key_1 == 'network':
            for key_2 in config['model']['network'].keys():
                logger.experiment.config['network_' + str(key_2)] = config['model']['network'][key_2]
        else:
            logger.experiment.config['model_' + str(key_1)] = config['model'][key_1]

    if config['log_video']:
        callbacks = [LogReplayVideoCallback(config['log_video_path'])]
    else:
        callbacks = []
    trainer = Trainer(**config['trainer'], logger=logger, callbacks=callbacks)
    trainer.fit(model)


if __name__ == "__main__":
    main()
