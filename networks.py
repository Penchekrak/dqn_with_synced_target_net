from typing import Tuple

from pl_bolts.models.rl.common.networks import CNN as SmallCNN
from torch import nn


class MediumCNN(nn.Module, SmallCNN):
    def __init__(self, input_shape: Tuple[int], n_actions: int):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super(MediumCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, n_actions)
        )
