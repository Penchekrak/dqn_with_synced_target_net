from functools import partial
from typing import Tuple

from pl_bolts.models.rl.common.networks import CNN as SmallCNN, MLP, DuelingCNN as SmallDuelingCNN
from torch import nn


class MediumCNN(SmallCNN):
    def __init__(self, input_shape: Tuple[int], n_actions: int):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super(SmallCNN, self).__init__()
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
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, n_actions)
        )


class LargeCNN(SmallCNN):
    def __init__(self, input_shape: Tuple[int], n_actions: int):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super(SmallCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, n_actions)
        )


SmallMLP = partial(MLP, hidden_size=128)
MediumMLP = partial(MLP, hidden_size=256)
LargeMLP = partial(MLP, hidden_size=512)


class MediumDuelingCNN(SmallDuelingCNN):
    def __init__(self, input_shape: Tuple[int], n_actions: int):
        super(SmallDuelingCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # advantage head
        self.head_adv = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

        # value head
        self.head_val = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1))


class LargeDuelingCNN(SmallDuelingCNN):
    def __init__(self, input_shape: Tuple[int], n_actions: int):
        super(SmallDuelingCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # advantage head
        self.head_adv = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                      nn.Linear(512, n_actions))

        # value head
        self.head_val = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                      nn.Linear(512, 1))
