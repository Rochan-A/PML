"""
Parts of this code was borrowed from:
https://github.com/facebookresearch/mbrl-lib
"""

import numpy as np
import torch

from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch
from torch.distributions import kl_divergence, Normal

import mbrl.types


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs, actions, sample=, rng=)`
    """

    def __init__(
        self,
        env: gym.Env,
        model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
    ):
        # whole model containing dynamics, backbone and context_enc
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
        context: torch.Tensor = None,
        c_embb: torch.Tensor = None,
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
            context (torch.Tensor or np.ndarray, Optional): the context to use.
            c_embb (torch.Tensor or np.ndarray, Optional): Directly use this context embedding
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
                context,
                c_embb=c_embb,
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
        initial_context: np.ndarray = None,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.
        
        NOTE: This is the default method to evaluate action sequences.
              Here, we compute the expected cummulative reward for each
              trajectory sequence. Rewards used are predicted by the forward
              dynamics model. Latent context are sampled i.i.d for each sequence.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.
            initial_context (np.ndarray, Optional): context for the rollout

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
                action_batch, model_state, initial_context, sample=True
            )
            rewards[terminated] = 0
            terminated |= dones.reshape(-1, 1)
            total_rewards += rewards

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(dim=1)

    def evaluate_action_sequences_kl(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
        initial_context: np.ndarray = None,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        NOTE: This method weights the expected cumulative reward from taking an
              action sequence by the KL divergence of the traced state-action
              sequence. The intuition is that we want to plan actions that do
              not deviate far from the current posterior over the latent context.
              This behaves as SAFE exploration!

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.
            initial_context (np.ndarray, Optional): context for the rollout

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        assert (
            len(action_sequences.shape) == 3
        )  # population_size, horizon, action_shape
        population_size, horizon, action_dim = action_sequences.shape
        state_dim = model_state["obs"].shape[-1]
        initial_obs_batch = np.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(np.float32)
        model_state = self.reset(initial_obs_batch, return_as_np=False)
        batch_size = initial_obs_batch.shape[0]
        total_rewards = torch.zeros(batch_size, 1).to(self.device)
        terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)

        # stack trajectory (state-action) rollouts
        generated_context_vec = torch.zeros(
            (batch_size, self.dynamics_model.context_enc.in_dim), dtype=torch.float32
        ).to(self.device)

        for time_step in range(horizon):
            actions_for_step = action_sequences[:, time_step, :]
            action_batch = torch.repeat_interleave(
                actions_for_step, num_particles, dim=0
            )

            # Store state, action
            generated_context_vec[:, time_step : time_step + state_dim] = model_state["obs"]
            generated_context_vec[:, time_step + state_dim : time_step + state_dim + action_dim] = action_batch

            _, rewards, dones, model_state = self.step(
                action_batch, model_state, initial_context, sample=True
            )

            rewards[terminated] = 0
            terminated |= dones.reshape(-1, 1)
            total_rewards += rewards

        # Get the posterior for the new generated context vec and past context vec 
        with torch.no_grad():
            _, mu, log_var = self.dynamics_model.context_enc.forward(generated_context_vec)
            new_dist = Normal(mu, torch.exp(log_var))

            if not isinstance(initial_context, torch.Tensor):
                initial_context = torch.tensor(initial_context).float().to(self.device)

            _, mu, log_var = self.dynamics_model.context_enc.forward(initial_context)
            old_dist = Normal(mu, torch.exp(log_var))

        # Weight the total reward for each trajectory by the KL divergence
        total_rewards *= self.dynamics_model.kl_weight \
                        * kl_divergence(old_dist, new_dist).mean(dim=1).reshape(-1, 1)

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(dim=1)

    def evaluate_action_sequences_greedy(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
        initial_context: np.ndarray = None,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        NOTE: Instead of sampling the posterior while evaluating each trajectory,
              we Monte Carlo sample the posterior and use a prediction loss
              over the context vector to greedily select the context embedding
              with the lowest prediction loss. The intuition is that we want to
              use an embedding that most closely follows the dynamics of the
              MDP. The rest of the evaluation function remains the same.
              This behaves as GREEDY exploration!

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.
            initial_context (np.ndarray, Optional): context for the rollout

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

        # MC sample {z'}_k ~ p(z | <initial_context>) & pick strongest sample
        z_sample = self.dynamics_model._maybe_strongest_mc_sample_context_enc(
            initial_context=initial_context, 
            state_sz=model_state["obs"].shape[-1],
            action_sz=action_dim
        )

        for time_step in range(horizon):
            actions_for_step = action_sequences[:, time_step, :]
            action_batch = torch.repeat_interleave(
                actions_for_step, num_particles, dim=0
            )

            # rather than passing the context vec, directly specify the context
            # embedding we found earlier to context-condition the dynamics model
            _, rewards, dones, model_state = self.step(
                actions=action_batch,
                model_state=model_state,
                context=None,
                c_embb=z_sample,
                sample=True
            )

            rewards[terminated] = 0
            terminated |= dones.reshape(-1, 1)
            total_rewards += rewards

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(dim=1)
