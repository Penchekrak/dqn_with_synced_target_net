import typing as tp
from collections import OrderedDict

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pl_bolts.losses.rl import dqn_loss
from pl_bolts.models.rl import DQN as FixedNetworkPLDQN
from pl_bolts.models.rl.double_dqn_model import DoubleDQN
from pl_bolts.models.rl.common.gym_wrappers import gym_make, MaxAndSkipEnv, FireResetEnv
from pl_bolts.models.rl.common.gym_wrappers import ImageToPyTorch, BufferWrapper, ScaledFloatFrame



class PLDQN(FixedNetworkPLDQN):
    def __init__(
            self,
            env: str,
            network: OmegaConf,
            eps_start: float = 1.0,
            eps_end: float = 0.02,
            eps_last_frame: int = 150000,
            sync_rate: int = 1000,
            gamma: float = 0.99,
            learning_rate: float = 1e-4,
            batch_size: int = 32,
            replay_size: int = 100000,
            warm_start_size: int = 10000,
            avg_reward_len: int = 100,
            min_episode_reward: int = -21,
            seed: int = 123,
            batches_per_epoch: int = 1000,
            n_steps: int = 1,
            **kwargs
    ):
        self.network = network
        super().__init__(env, eps_start, eps_end, eps_last_frame, sync_rate, gamma, learning_rate, batch_size,
                         replay_size, warm_start_size, avg_reward_len, min_episode_reward, seed, batches_per_epoch,
                         n_steps, **kwargs)

    def build_networks(self) -> None:
        self.net = instantiate(self.network, self.obs_shape, self.n_actions)
        self.target_net = instantiate(self.network, self.obs_shape, self.n_actions)


class PLDoubleDQN(DoubleDQN):
    def __init__(
            self,
            env: str,
            network: OmegaConf,
            eps_start: float = 1.0,
            eps_end: float = 0.02,
            eps_last_frame: int = 150000,
            sync_rate: int = 1000,
            gamma: float = 0.99,
            learning_rate: float = 1e-4,
            batch_size: int = 32,
            replay_size: int = 100000,
            warm_start_size: int = 10000,
            avg_reward_len: int = 100,
            min_episode_reward: int = -21,
            seed: int = 123,
            batches_per_epoch: int = 1000,
            n_steps: int = 1,
            **kwargs
    ):
        self.network = network
        super().__init__(env, eps_start, eps_end, eps_last_frame, sync_rate, gamma, learning_rate, batch_size,
                         replay_size, warm_start_size, avg_reward_len, min_episode_reward, seed, batches_per_epoch,
                         n_steps, **kwargs)

    def build_networks(self) -> None:
        self.net = instantiate(self.network, self.obs_shape, self.n_actions)
        self.target_net = instantiate(self.network, self.obs_shape, self.n_actions)


class SyncedTargetNetworkDQN(PLDQN):
    def build_networks(self) -> None:
        self.net = instantiate(self.network, self.obs_shape, self.n_actions)
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


class SyncedTargetNetworkDoubleDQN(PLDoubleDQN):
    def build_networks(self) -> None:
        self.net = instantiate(self.network, self.obs_shape, self.n_actions)
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
                # "episodes": self.done_episodes,
                # "episode_steps": self.total_episode_steps[-1],
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
            }
        )


class RAMPLDQN(PLDQN):
    @staticmethod
    def make_environment(env_name: str, seed=None):
        env = gym_make(env_name)
        env = MaxAndSkipEnv(env)
        env = FireResetEnv(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 4)
        return ScaledFloatFrame(env)

        if seed:
            env.seed(seed)