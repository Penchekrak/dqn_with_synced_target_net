import typing as tp
from collections import OrderedDict

import torch
from pl_bolts.losses.rl import dqn_loss
from pl_bolts.models.rl import DQN as PLDQN
from pl_bolts.models.rl.common.networks import CNN


class SyncedTargetNetworkDQN(PLDQN):
    def build_networks(self) -> None:
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = self.net

    def training_step(self, batch: tp.Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.
        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net, self.gamma)

        if self._use_dp_or_ddp2(self.trainer):
            loss = loss.unsqueeze(0)

        # No sync step for target network (Soft update of target network)
        # if self.global_step % self.sync_rate == 0:
        #     self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "train_loss": loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
            }
        )
