# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import warnings
from typing import Any, List, Optional, Sequence, Sized, Tuple, Type, Union

import numpy as np

from mbrl.types import TransitionBatch


def create_replay_buffer(cfg, obs_shape, act_shape, rng=None):
    dataset_size = (
        cfg.algorithm.get("dataset_size", None) if "algorithm" in cfg else None
    )
    if not dataset_size:
        dataset_size = cfg.overrides.num_steps
    maybe_max_trajectory_len = None

    if cfg.overrides.trial_length is None:
        raise ValueError(
            "cfg.overrides.trial_length must be set when "
            "collect_trajectories==True."
        )
    maybe_max_trajectory_len = cfg.overrides.trial_length

    replay_buffer = ReplayBuffer(
        dataset_size,
        obs_shape,
        act_shape,
        rng=rng,
        max_trajectory_length=maybe_max_trajectory_len,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        replay_buffer.load(str(load_dir))

    return replay_buffer


class ReplayBuffer:
    """A replay buffer with support for training/validation iterators and ensembles.

    This buffer can be pushed to and sampled from as a typical replay buffer.

    Args:
        capacity (int): the maximum number of transitions that the buffer can store.
            When the capacity is reached, the contents are overwritten in FIFO fashion.
        obs_shape (Sequence of ints): the shape of the observations to store.
        action_shape (Sequence of ints): the shape of the actions to store.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
        max_trajectory_length (int, optional): if given, indicates that trajectory
            information should be stored and that trajectories will be at most this
            number of steps. Defaults to ``None`` in which case no trajectory
            information will be kept. The buffer will keep trajectory information
            automatically using the done value when calling :meth:`add`.

    .. warning::
        When using ``max_trajectory_length`` it is the user's responsibility to ensure
        that trajectories are stored continuously in the replay buffer.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        obs_type: Type = np.float32,
        action_type: Type = np.float32,
        reward_type: Type = np.float32,
        rng: Optional[np.random.Generator] = None,
        max_trajectory_length: Optional[int] = None,
    ):
        self.cur_idx = 0
        self.capacity = capacity
        self.num_stored = 0

        self.trajectory_indices: Optional[List[Tuple[int, int]]] = None
        if max_trajectory_length:
            self.trajectory_indices = []
            capacity += max_trajectory_length
        # TODO replace all of these with a transition batch
        self.obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.action = np.empty((capacity, *action_shape), dtype=action_type)
        self.reward = np.empty(capacity, dtype=reward_type)
        self.done = np.empty(capacity, dtype=bool)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._start_last_trajectory = 0

    @property
    def stores_trajectories(self) -> bool:
        return self.trajectory_indices is not None

    @staticmethod
    def _check_overlap(segment1: Tuple[int, int], segment2: Tuple[int, int]) -> bool:
        s1, e1 = segment1
        s2, e2 = segment2
        return (s1 <= s2 < e1) or (s1 < e2 <= e1)

    def remove_overlapping_trajectories(self, new_trajectory: Tuple[int, int]):
        cnt = 0
        for traj in self.trajectory_indices:
            if self._check_overlap(new_trajectory, traj):
                cnt += 1
            else:
                break
        for _ in range(cnt):
            self.trajectory_indices.pop(0)

    def _trajectory_bookkeeping(self, done: bool):
        self.cur_idx += 1
        if self.num_stored < self.capacity:
            self.num_stored += 1
        if self.cur_idx >= self.capacity:
            self.num_stored = max(self.num_stored, self.cur_idx)
        if done:
            self.close_trajectory()
        else:
            partial_trajectory = (self._start_last_trajectory, self.cur_idx + 1)
            self.remove_overlapping_trajectories(partial_trajectory)
        if self.cur_idx >= len(self.obs):
            warnings.warn(
                "The replay buffer was filled before current trajectory finished. "
                "The history of the current partial trajectory will be discarded. "
                "Make sure you set `max_trajectory_length` to the appropriate value "
                "for your problem."
            )
            self._start_last_trajectory = 0
            self.cur_idx = 0
            self.num_stored = len(self.obs)

    def close_trajectory(self):
        new_trajectory = (self._start_last_trajectory, self.cur_idx)
        self.remove_overlapping_trajectories(new_trajectory)
        self.trajectory_indices.append(new_trajectory)

        if self.cur_idx - self._start_last_trajectory > (len(self.obs) - self.capacity):
            warnings.warn(
                "A trajectory was saved with length longer than expected. "
                "Unexpected behavior might occur."
            )

        if self.cur_idx >= self.capacity:
            self.cur_idx = 0
        self._start_last_trajectory = self.cur_idx

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Adds a transition (s, a, s', r, done) to the replay buffer.

        Args:
            obs (np.ndarray): the observation at time t.
            action (np.ndarray): the action at time t.
            next_obs (np.ndarray): the observation at time t + 1.
            reward (float): the reward at time t + 1.
            done (bool): a boolean indicating whether the episode ended or not.
        """
        self.obs[self.cur_idx] = obs
        self.next_obs[self.cur_idx] = next_obs
        self.action[self.cur_idx] = action
        self.reward[self.cur_idx] = reward
        self.done[self.cur_idx] = done

        if self.trajectory_indices is not None:
            self._trajectory_bookkeeping(done)
        else:
            self.cur_idx = (self.cur_idx + 1) % self.capacity
            self.num_stored = min(self.num_stored + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int): the number of samples required.

        Returns:
            (tuple): the sampled values of observations, actions, next observations, rewards
            and done indicators, as numpy arrays, respectively. The i-th transition corresponds
            to (obs[i], act[i], next_obs[i], rewards[i], dones[i]).
        """
        indices = self._rng.choice(self.num_stored, size=batch_size)
        return self._batch_from_indices(indices)

    def sample_trajectory(self) -> Optional[TransitionBatch]:
        """Samples a full trajectory and returns it as a batch.

        Returns:
            (tuple): A tuple with observations, actions, next observations, rewards
            and done indicators, as numpy arrays, respectively; these will correspond
            to a full trajectory. The i-th transition corresponds
            to (obs[i], act[i], next_obs[i], rewards[i], dones[i])."""
        if self.trajectory_indices is None or len(self.trajectory_indices) == 0:
            return None
        idx = self._rng.choice(len(self.trajectory_indices))
        indices = np.arange(
            self.trajectory_indices[idx][0], self.trajectory_indices[idx][1]
        )
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices: Sized) -> TransitionBatch:
        obs = self.obs[indices]
        next_obs = self.next_obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        done = self.done[indices]

        return TransitionBatch(obs, action, next_obs, reward, done)

    def __len__(self):
        return self.num_stored

    def save(self, save_dir: Union[pathlib.Path, str]):
        """Saves the data in the replay buffer to a given directory.

        Args:
            save_dir (str): the directory to save the data to. File name will be
                replay_buffer.npz.
        """
        path = pathlib.Path(save_dir) / "replay_buffer.npz"
        np.savez(
            path,
            obs=self.obs[: self.num_stored],
            next_obs=self.next_obs[: self.num_stored],
            action=self.action[: self.num_stored],
            reward=self.reward[: self.num_stored],
            done=self.done[: self.num_stored],
        )

    def load(self, load_dir: Union[pathlib.Path, str]):
        """Loads transition data from a given directory.

        Args:
            load_dir (str): the directory where the buffer is stored.
        """
        path = pathlib.Path(load_dir) / "replay_buffer.npz"
        data = np.load(path)
        num_stored = len(data["obs"])
        self.obs[:num_stored] = data["obs"]
        self.next_obs[:num_stored] = data["next_obs"]
        self.action[:num_stored] = data["action"]
        self.reward[:num_stored] = data["reward"]
        self.done[:num_stored] = data["done"]
        self.num_stored = num_stored
        self.cur_idx = self.num_stored % self.capacity

    def get_all(self, shuffle: bool = False) -> TransitionBatch:
        """Returns all data stored in the replay buffer.

        Args:
            shuffle (int): set to ``True`` if the data returned should be in random order.
            Defaults to ``False``.
        """
        if shuffle:
            permutation = self._rng.permutation(self.num_stored)
            return self._batch_from_indices(permutation)
        else:
            return TransitionBatch(
                self.obs[: self.num_stored],
                self.action[: self.num_stored],
                self.next_obs[: self.num_stored],
                self.reward[: self.num_stored],
                self.done[: self.num_stored],
            )

    def get_iterators(
        self,
        batch_size: int,
        val_ratio: float,
        train_ensemble: bool = False,  # noqa
        ensemble_size: Optional[int] = None,
        shuffle_each_epoch: bool = True,
        bootstrap_permutes: bool = False,
    ):
        """Returns training/validation iterators for the data in the replay buffer.

        .. deprecated:: v0.1.2
           Use :func:`mbrl.util.common.get_basic_buffer_iterators`.


        Args:
            batch_size (int): the batch size for the iterators.
            val_ratio (float): the proportion of data to use for validation. If 0., the
                validation buffer will be set to ``None``.
            train_ensemble (bool): if ``True``, the training iterator will be and
                instance of :class:`BootstrapIterator`. Defaults to ``False``.
            ensemble_size (int): the size of the ensemble being trained. Must be
                provided if ``train_ensemble == True``.
            shuffle_each_epoch (bool): if ``True``, the iterator will shuffle the
                order each time a loop starts. Otherwise the iteration order will
                be the same. Defaults to ``True``.
            bootstrap_permutes (bool): if ``True``, the bootstrap iterator will create
                the bootstrap data using permutations of the original data. Otherwise
                it will use sampling with replacement. Defaults to ``False``.

        """
        warnings.warn(
            "ReplayBuffer.get_iterators() is deprecated and will be removed "
            " starting on v0.2.0. Use mbrl.util.common.get_basic_iterators() "
            " instead."
        )
        from mbrl.util.common import get_basic_buffer_iterators

        return get_basic_buffer_iterators(
            self,
            batch_size,
            val_ratio,
            1 if ensemble_size is None else ensemble_size,
            shuffle_each_epoch,
            bootstrap_permutes,
        )

    @property
    def rng(self) -> np.random.Generator:
        return self._rng