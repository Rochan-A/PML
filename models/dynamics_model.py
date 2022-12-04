import numpy as np
import torch

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch

import mbrl.types


class RollingHistoryContext:
    def __init__(self, K, state_sz, action_sz) -> None:
        self.K = K
        
        self.state_sz = state_sz
        self.action_sz = action_sz
        self.default_sz = (state_sz+action_sz)*K
        self.store = None
        self.prev_st = None

    def append(self, state, action):
        if self.prev_st is None:
            self.prev_st = state
        else:
            state = state - self.prev_st

        k = np.concatenate([state, action], axis=0)
        if self.store is None:
            self.store = np.repeat(k, self.K)
        else:
            self.store = np.roll(self.store, -k.shape[0])
            self.store[-k.shape[0]:] = k

    def reset(self):
        self.store = None
        self.prev_st = None


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs, actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

    def __init__(
        self,
        env: gym.Env,
        model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True

    def reset(
        self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        model_state = self.dynamics_model.reset(
            initial_obs_batch.astype(np.float32), rng=self._rng
        )
        self._return_as_np = return_as_np
        return model_state if model_state is not None else {}

    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            (
                next_observs,
                pred_rewards,
                pred_terminals,
                next_model_state,
            ) = self.dynamics_model.sample(
                actions,
                model_state,
                deterministic=not sample,
                rng=self._rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs)

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )

            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, next_model_state

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        assert (
            len(action_sequences.shape) == 3
        )  # population_size, horizon, action_shape
        population_size, horizon, action_dim = action_sequences.shape
        initial_obs_batch = np.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(np.float32)
        model_state = self.reset(initial_obs_batch, return_as_np=False)
        batch_size = initial_obs_batch.shape[0]
        total_rewards = torch.zeros(batch_size, 1).to(self.device)
        terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
        for time_step in range(horizon):
            actions_for_step = action_sequences[:, time_step, :]
            action_batch = torch.repeat_interleave(
                actions_for_step, num_particles, dim=0
            )
            _, rewards, dones, model_state = self.step(
                action_batch, model_state, sample=True
            )
            rewards[terminated] = 0
            terminated |= dones
            total_rewards += rewards

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(dim=1)



if __name__=='__main__':
    import sys
    sys.path.append("..")  # Adds higher directory to python modules path.

    import math
    import yaml
    from easydict import EasyDict
    import omegaconf
    from transitionreward import create_model

    from envs import ContexualEnv
    from envs.CartpoleBalance import CartPoleEnv_template

    with open('../configs/cartpole.yaml') as f:
        config = yaml.safe_load(f)
        config['device'] = torch.device('cpu')
    config = EasyDict(config)

    env_fam = ContexualEnv(config)
    env, context = env_fam.reset(train=True)

    trial_length = 200
    num_trials = 10
    ensemble_size = 5

    context_cfg = {
        'state_sz': env.observation_space.shape[0],
        'action_sz': 1,
        'hidden_dim': 128,
        'hidden_layers': 1,
        'out_dim': 32,
        'history_size': 16
    }

    backbone_cfg = {
        'state_sz': env.observation_space.shape[0],
        'action_sz': 1,
        'hidden_dim': 128,
        'hidden_layers': 1,
        'out_dim': 32
    }

    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the
    # environment information
    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "model":
            {
                "_target_": "mbrl.models.GaussianMLP",
                "device": 'cpu',
                "num_layers": 3,
                "ensemble_size": ensemble_size,
                "hid_size": 128,
                "in_size": context_cfg['out_dim']+backbone_cfg['out_dim'],
                "out_size": "???",
                "deterministic": False,
                "propagation_method": "fixed_model",
                # can also configure activation function for GaussianMLP
                "activation_fn_cfg": {
                    "_target_": "torch.nn.LeakyReLU",
                    "negative_slope": 0.01
                }
            }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05,
        }
    }
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    
    def term_fn(action, state):
        x, x_dot, theta, theta_dot = state
        done = x < -2.4 \
               or x > 2.4 \
               or theta < -(12 * 2 * math.pi / 360) \
               or theta > (12 * 2 * math.pi / 360)
        done = bool(done)
        return done

    reward_fn = None

    dynamics_model = create_model(cfg, context_cfg, backbone_cfg)
    model_env = ModelEnv(env, dynamics_model, term_fn, reward_fn)

    print(model_env)
