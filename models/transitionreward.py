"""
Parts of this code was borrowed from:
https://github.com/facebookresearch/mbrl-lib
"""

import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import hydra

import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math

from mbrl.models.model import Ensemble, Model
from .base import StateActionEncoder, ContextEncoder


def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return (
        -0.5 * torch.log(2 * np.pi * sigma**2)
        - (1 / (2 * sigma**2)) * (y - mu) ** 2
    )


def elbo(y_pred, mu, log_var):
    # prior probability of y_pred
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.0)))

    # variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)

    # by taking the mean we approximate the expectation
    return (log_prior - log_p_q).mean()


def det_loss(y_pred, mu, log_var):
    # Neg of elbo without reconstruction loss term
    return -elbo(y_pred, mu, log_var)


def det_loss_kl(mu, log_var):
    # KL( p(z) || q(z) )
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
    return kl_divergence.sum()


class TransitionRewardModel(Model):
    """
    Model inputs/outputs will be consistent with

        [pred_obs_{t+1}, pred_rewards_{t+1} (optional)] = model([obs_t, action_t, context]).

    To use with :class:mbrl.models.ModelEnv`, the wrapped model must define methods
    ``reset_1d`` and ``sample_1d``.
    """

    def __init__(
        self,
        model: Model,
        context_cfg: dict,
        stateaction_cfg: dict,
        eval_cfg: dict,
        use_context=False,
        target_is_delta: bool = True,
        normalize: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        num_elites: Optional[int] = None,
    ):
        super().__init__(model.device)

        self.use_context = use_context

        # GaussianMLP model
        self.model = model
        self.context_enc = ContextEncoder(**context_cfg).to(self.model.device)
        self.stateaction_enc = StateActionEncoder(**stateaction_cfg).to(
            self.model.device
        )

        self.context_enc.load('/workspace/rochan/model-based/PML/saves/CartpoleSwingUp/raw/seed_0/2023-01-06_09-57')

        # configs for evaluating action sequences
        self.eval_cfg = eval_cfg

        # we only normalize the current state, action.
        # Context encoder is using state delta so we do not normalize it
        self.input_normalizer: Optional[mbrl.util.math.Normalizer] = None
        if normalize:
            in_size = self.stateaction_enc.in_dim
            self.input_normalizer = mbrl.util.math.Normalizer(
                in_size,
                self.model.device,
                dtype=torch.double if normalize_double_precision else torch.float,
            )

        self.device = self.model.device
        self.learned_rewards = learned_rewards
        self.target_is_delta = target_is_delta

        self.num_elites = num_elites
        if not num_elites and isinstance(self.model, Ensemble):
            self.num_elites = self.model.num_members

    def _get_model_input(self, obs, action, context=None) -> torch.Tensor:
        obs = model_util.to_tensor(obs).to(self.device)
        action = model_util.to_tensor(action).to(self.device)
        model_in = torch.cat([obs, action], dim=obs.ndim - 1)
        if self.input_normalizer:
            # Normalizer lives on device
            model_in = self.input_normalizer.normalize(model_in).float().to(self.device)
        if context is not None:
            context = model_util.to_tensor(context).float().to(self.device)
            model_in = torch.cat([context, model_in], dim=obs.ndim - 1)
        return model_in

    def _process_batch(
        self, batch: mbrl.types.TransitionBatch, _as_float: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bb = batch.astuple()
        if len(bb) == 5:
            obs, action, next_obs, reward, _ = bb
            context = None
        else:
            obs, action, next_obs, reward, _, context = bb
        if self.target_is_delta:
            target_obs = next_obs - obs
        else:
            target_obs = next_obs
        target_obs = model_util.to_tensor(target_obs).to(self.device)

        model_in = self._get_model_input(obs, action, context)
        if self.learned_rewards:
            reward = model_util.to_tensor(reward).to(self.device).unsqueeze(reward.ndim)
            target = torch.cat([target_obs, reward], dim=obs.ndim - 1)
        else:
            target = target_obs
        return model_in.float(), target.float()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Calls forward method of base model with the given input and args."""
        return self.model.forward(x, *args, **kwargs)

    def update_normalizer(self, batch: mbrl.types.TransitionBatch):
        """Updates the normalizer statistics using the batch of transition data.

        The normalizer will compute mean and standard deviation the obs and action in
        the transition. If an observation processing function has been provided, it will
        be called on ``obs`` before updating the normalizer.

        Args:
            batch (:class:`mbrl.types.TransitionBatch`): The batch of transition data.
                Only obs and action will be used, since these are the inputs to the model.
        """
        if self.input_normalizer is None:
            return
        obs, action = batch.obs, batch.act
        if obs.ndim == 1:
            obs = obs[None, :]
            action = action[None, :]
        model_in_np = np.concatenate([obs, action], axis=obs.ndim - 1)
        self.input_normalizer.update_stats(model_in_np)

    def loss(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the model loss over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.loss()` on them and returns the value and the metadata
        as returned by the model.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor and optional dict): as returned by `model.loss().`
        """
        assert target is None
        model_in, target = self._process_batch(batch)

        if self.use_context:
            # Split to context, state, action
            context = model_in[..., : self.context_enc.in_dim]
            s_t = model_in[..., self.context_enc.in_dim : -1]
            a_t = model_in[..., -1:]
            s = context.shape

            with torch.no_grad():
                c_embb, c_mu, c_log_var = self.context_enc.forward(
                    context.reshape(-1, context.shape[-1])
                )
                if len(s) == 3:
                    c_embb = c_embb.reshape(s[0], s[1], -1)
                    c_mu = c_mu.reshape(s[0], s[1], -1)
                    c_log_var = c_log_var.reshape(s[0], s[1], -1)
                else:
                    c_embb = c_embb.reshape(s[0], -1)
                    c_mu = c_mu.reshape(s[0], -1)
                    c_log_var = c_log_var.reshape(s[0], -1)

                # Context encoder loss based on:
                # Generalized Hidden Parameter MDPs Transferable Model-based RL in a Handful of Trials
                # https://arxiv.org/abs/2002.03072
                # loss += det_loss_kl(c_mu, c_log_var)

            # forward pass over the state-action encoder & dynamics model
            b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
            model_in = torch.cat([c_embb, b_embb], dim=-1)
            loss += self.model.loss(model_in, target=target)
        else:
            s_t = model_in[..., :-1]
            a_t = model_in[..., -1:]
            # forward pass over the backbone encoder
            b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
            loss += self.model.loss(b_embb, target=target)
        return loss

    def update(
        self,
        batch: mbrl.types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Updates the model given a batch of transitions and an optimizer.

        Args:
            batch (transition batch): a batch of transition to train the model.
            optimizer (torch optimizer): the optimizer to use to update the model.

        Returns:
            (tensor and optional dict): as returned by `model.loss().`
        """
        assert target is None
        model_in, target = self._process_batch(batch)

        self.model.train()
        optimizer.zero_grad()

        loss = 0
        if self.use_context:
            self.context_enc.train()
            self.stateaction_enc.train()

            # Split to context, state, action
            context = model_in[..., : self.context_enc.in_dim]
            s_t = model_in[..., self.context_enc.in_dim : -1]
            a_t = model_in[..., -1:]
            s = context.shape
            context = context.reshape(-1, context.shape[-1])

            # Monte Carlo sample from p(c), and avg loss -- context encoder
            loss_and_maybe_meta = []
            for _ in range(self.eval_cfg.mc_update):
                with torch.no_grad():
                    c_embb, c_mu, c_log_var = self.context_enc.forward(
                        context.detach().clone()
                    )
                    if len(s) == 3:
                        c_embb = c_embb.reshape(s[0], s[1], -1)
                        c_mu = c_mu.reshape(s[0], s[1], -1)
                        c_log_var = c_log_var.reshape(s[0], s[1], -1)
                    else:
                        c_embb = c_embb.reshape(s[0], -1)
                        c_mu = c_mu.reshape(s[0], -1)
                        c_log_var = c_log_var.reshape(s[0], -1)

                    # Context encoder loss based on:
                    # Generalized Hidden Parameter MDPs Transferable Model-based RL in a Handful of Trials
                    # https://arxiv.org/abs/2002.03072
                    # loss += det_loss_kl(c_mu, c_log_var)

                # forward pass over the state-action encoder & dynamics model
                b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
                model_in = torch.cat([c_embb, b_embb], dim=-1)

                # Update the model using backpropagation with given input and target tensors.
                # if self.deterministic: returns self._mse_loss(model_in, target), {}
                # else returns self._nll_loss(model_in, target), {}
                loss_ = self.model.loss(model_in, target)[0]
                loss_and_maybe_meta.append(loss_.reshape(1, -1))
            loss_and_maybe_meta = torch.mean(
                torch.cat(loss_and_maybe_meta, dim=-1), dim=-1
            )[0]
        else:
            s_t = model_in[..., :-1]
            a_t = model_in[..., -1:]
            # forward pass over the backbone encoder
            b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
            loss_and_maybe_meta = self.model.loss(b_embb, target)[0]

        if isinstance(loss_and_maybe_meta, tuple):
            # TODO - v0.2.0 remove this back-compatibility logic
            loss += cast(torch.Tensor, loss_and_maybe_meta[0])
            meta = cast(Dict[str, Any], loss_and_maybe_meta[1])
            loss.backward()

            if meta is not None:
                with torch.no_grad():
                    grad_norm = 0.0
                    for p in list(
                        filter(lambda p: p.grad is not None, self.parameters())
                    ):
                        grad_norm += p.grad.data.norm(2).item() ** 2
                    meta["grad_norm"] = grad_norm
            optimizer.step()
            ret = [loss.item(), meta]

        else:
            loss += loss_and_maybe_meta
            loss.backward()
            optimizer.step()
            ret = [loss.item()]

        return ret

    def eval_score(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Evaluates the model score over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.eval_score()` on them and returns the value.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tensor): as returned by `model.eval_score().`
        """
        assert target is None
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            if self.use_context:
                # Split to context, state, action
                context = model_in[..., : self.context_enc.in_dim]
                s_t = model_in[..., self.context_enc.in_dim : -1]
                a_t = model_in[..., -1:]
                s = context.shape

                c_embb, _, _ = self.context_enc.forward(
                    context.reshape(-1, context.shape[-1])
                )
                if len(s) == 3:
                    c_embb = c_embb.reshape(s[0], s[1], -1)
                else:
                    c_embb = c_embb.reshape(s[0], -1)
                b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
                model_in = torch.cat([c_embb, b_embb], dim=-1)
                return self.model.eval_score(model_in, target=target)
            else:
                s_t = model_in[..., :-1]
                a_t = model_in[..., -1:]
                # forward pass over the state-action encoder & dynamics model
                b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
                return self.model.eval_score(b_embb, target=target)

    def get_output_and_targets(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns the model output and the target tensors given a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.forward()` on them and returns the value.
        No gradient information will be kept.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tuple(tensor), tensor): the model outputs and the target for this batch.
        """
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            if self.use_context:
                # Split to context, state, action
                context = model_in[..., : self.context_enc.in_dim]
                s_t = model_in[..., self.context_enc.in_dim : -1]
                a_t = model_in[..., -1:]
                s = context.shape

                c_embb, _, _ = self.context_enc.forward(
                    context.reshape(-1, context.shape[-1])
                )
                if len(s) == 3:
                    c_embb = c_embb.reshape(s[0], s[1], -1)
                else:
                    c_embb = c_embb.reshape(s[0], -1)
                b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
                model_in = torch.cat([c_embb, b_embb], dim=-1)
                output = self.model.forward(model_in)
                return output, target
            else:
                s_t = model_in[..., :-1]
                a_t = model_in[..., -1:]
                b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
                output = self.model.forward(b_embb)
                return output, target

    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        initial_context: np.ndarray = None,
        c_embb: np.ndarray = None,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """Samples next observations and rewards from the underlying 1-D model.

        This wrapper assumes that the underlying model's sample method returns a tuple
        with just one tensor, which concatenates next_observation and reward.

        Args:
            act (tensor): the action at.
            model_state (dict(str, tensor)): the model state st.
            initial_context (np.ndarray, Optional): context vector
            c_embb (np.ndarray, Optional): context embedding
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (random number generator): a rng to use for sampling.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        if c_embb is not None:
            return self.sample_c_embb(act, model_state, c_embb, deterministic, rng)

        obs = model_util.to_tensor(model_state["obs"]).to(self.device)

        if initial_context is not None:
            if len(initial_context.shape) == 1 and len(model_state["obs"].shape) == 2:
                initial_context = np.repeat(
                    initial_context[None, :], model_state["obs"].shape[0], axis=0
                ).astype(np.float32)

        model_in = self._get_model_input(model_state["obs"], act, initial_context)
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method sample_1d"
            )

        if self.use_context:
            # Split to context, state, action
            context = model_in[..., : self.context_enc.in_dim]
            s_t = model_in[..., self.context_enc.in_dim : -1]
            a_t = model_in[..., -1:]
            s = context.shape

            c_embb, _, _ = self.context_enc.forward(
                context.reshape(-1, context.shape[-1])
            )
            if len(s) == 3:
                c_embb = c_embb.reshape(s[0], s[1], -1)
            else:
                c_embb = c_embb.reshape(s[0], -1)
            b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
            model_in = torch.cat([c_embb, b_embb], dim=-1)
        else:
            s_t = model_in[..., :-1]
            a_t = model_in[..., -1:]

            model_in = self.stateaction_enc.joint_embb(s_t, a_t)

        preds, next_model_state = self.model.sample_1d(
            model_in, model_state, rng=rng, deterministic=deterministic
        )
        next_observs = preds[:, :-1] if self.learned_rewards else preds
        if self.target_is_delta:
            tmp_ = next_observs + obs
            next_observs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None
        next_model_state["obs"] = next_observs
        return next_observs, rewards, None, next_model_state

    def sample_c_embb(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        c_embb: np.ndarray,
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """Samples next observations and rewards from the underlying 1-D model.

        This wrapper assumes that the underlying model's sample method returns a tuple
        with just one tensor, which concatenates next_observation and reward.

        Args:
            act (tensor): the action at.
            model_state (dict(str, tensor)): the model state st.
            c_embb (np.ndarray): context embedding
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (random number generator): a rng to use for sampling.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        obs = model_util.to_tensor(model_state["obs"]).to(self.device)

        if len(obs.shape) == 2:
            c_embb = np.repeat(c_embb, model_state["obs"].shape[0], axis=0).astype(
                np.float32
            )
            c_embb = model_util.to_tensor(c_embb).float().to(self.device)

        # Dont need to pass context because we already have the embedding
        model_in = self._get_model_input(model_state["obs"], act)
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method sample_1d"
            )

        s_t = model_in[..., :-1]
        a_t = model_in[..., -1:]
        b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
        model_in = torch.cat([c_embb, b_embb], dim=-1)

        preds, next_model_state = self.model.sample_1d(
            model_in, model_state, rng=rng, deterministic=deterministic
        )
        next_observs = preds[:, :-1] if self.learned_rewards else preds
        if self.target_is_delta:
            tmp_ = next_observs + obs
            next_observs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None
        next_model_state["obs"] = next_observs
        return next_observs, rewards, None, next_model_state

    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Calls reset on the underlying model.

        Args:
            obs (tensor): the observation from which the trajectory will be
                started. The actual value is ignore, only the shape is used.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (dict(str, tensor)): the model state necessary to continue the simulation.
        """
        if not hasattr(self.model, "reset_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method reset_1d"
            )
        obs = model_util.to_tensor(obs).to(self.device)
        model_state = {"obs": obs}
        model_state.update(self.model.reset_1d(obs, rng=rng))
        return model_state

    def save(self, save_dir: Union[str, pathlib.Path]):
        self.model.save(pathlib.Path(save_dir) / "gaussian_mlp.pth")
        self.context_enc.save(save_dir)
        self.stateaction_enc.save(save_dir)
        if self.input_normalizer:
            self.input_normalizer.save(save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        self.model.load(pathlib.Path(load_dir) / "gaussian_mlp.pth")
        self.context_enc.load(load_dir)
        self.stateaction_enc.load(load_dir)
        if self.input_normalizer:
            self.input_normalizer.load(load_dir)

    def set_elite(self, elite_indices: Sequence[int]):
        self.model.set_elite(elite_indices)

    def __len__(self):
        return len(self.model)

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        if isinstance(self.model, Ensemble):
            self.model.set_propagation_method(propagation_method)

    def _context_vec_delta_to_abs_state_action(self, context_vec, state_sz, action_sz):
        """Convert context vector to states and actions array"""
        states = [context_vec[:state_sz]]
        actions = [context_vec[state_sz : state_sz + action_sz]]
        for i in range(
            state_sz + action_sz - 1, context_vec.shape[0] - 1, state_sz + action_sz
        ):
            states.append(states[-1] + context_vec[i : i + state_sz])
            actions.append(context_vec[i + state_sz : i + state_sz + action_sz])
        return np.stack(states, axis=0), np.concatenate(actions, axis=-1)

    def _pick_strongest_sample(self, z_samples, states, actions):
        """Pick strongest latent context embedding sample based on prediction
        error
        """
        scores = np.zeros((self.eval_cfg.mc_samples,))
        with torch.no_grad():
            # contruct a batch predictor over the past states and actions
            model_in = self._get_model_input(states, actions[:, None])
            s_t = model_in[..., :-1].float()
            a_t = model_in[..., -1:].float()
            for mc_run in range(self.eval_cfg.mc_samples):
                c_embb = (
                    model_util.to_tensor(z_samples[mc_run, :])
                    .float()
                    .to(self.device)
                    .repeat(s_t.shape[0], 1)
                )
                b_embb = self.stateaction_enc.joint_embb(s_t, a_t)
                model_in = torch.cat([c_embb, b_embb], dim=-1)
                next_state_delta_mu, _ = self.model.forward(model_in, use_propagation=False)
                means = torch.mean(next_state_delta_mu, dim=0).cpu().numpy()

                # construct the target
                target = states[1:] - states[:1]
                # exclude the last prediction, reward for each prediction
                scores[mc_run] = np.sum((target - means[:-1, :-1]) ** 2) / (
                    (len(states) - 1) * states.shape[-1]
                )
        return np.argmin(scores)

    def _maybe_strongest_mc_sample_context_enc(
        self, initial_context, state_sz, action_sz
    ):
        """Monte-Carlo sample <self.mc_sample>'s from P(C | <initial_context>)"""
        states, actions = self._context_vec_delta_to_abs_state_action(
            initial_context, state_sz, action_sz
        )

        # MC sample the posterior
        initial_context = model_util.to_tensor(initial_context).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            dist = self.context_enc.dist(initial_context)
        z_samples = dist.sample(sample_shape=(self.eval_cfg.mc_samples,)).cpu().numpy()

        # Pick the strongest posterior sample
        strongest_idx = self._pick_strongest_sample(z_samples, states, actions)
        return z_samples[strongest_idx, :]


def create_model(
    cfg,
    context_cfg,
    stateaction_cfg,
    eval_cfg,
    use_context,
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """Creates a transition-reward model from a given configuration.

    The configuration should be structured as follows::
        -cfg
          -dynamics_model
            -model
              -_target_ (str): model Python class
              -in_size (int): input size
              -out_size (int, optional): output size
              -model_arg_1
               ...
              -model_arg_n
          -algorithm
            -learned_rewards (bool): whether rewards should be learned or not
            -target_is_delta (bool): to be passed to the dynamics model wrapper
            -normalize (bool): to be passed to the dynamics model wrapper
          -overrides
            -no_delta_list (list[int], optional): to be passed to the dynamics model wrapper
            -obs_process_fn (str, optional): a Python function to pre-process observations
            -num_elites (int, optional): number of elite members for ensembles

    The model will be instantiated using :func:`hydra.utils.instantiate` function.

    Args:
        cfg (omegaconf.DictConfig): the configuration to read.
        model_dir (str or pathlib.Path): If provided, the model will attempt to load its
            weights and normalization information from "model_dir / model.pth" and
            "model_dir / env_stats.pickle", respectively.

    Returns:
        (:class:`TransitionRewardModel`): the model created.

    """
    model_cfg = cfg.dynamics_model.model
    if model_cfg.get("out_size", None) is None:
        model_cfg.out_size = context_cfg.state_sz + int(cfg.algorithm.learned_rewards)

    # Now instantiate the model
    model = hydra.utils.instantiate(cfg.dynamics_model.model)

    dynamics_model = TransitionRewardModel(
        model=model,
        context_cfg=context_cfg,
        stateaction_cfg=stateaction_cfg,
        eval_cfg=eval_cfg,
        use_context=use_context,
        target_is_delta=cfg.algorithm.target_is_delta,
        normalize=cfg.algorithm.normalize,
        normalize_double_precision=cfg.algorithm.get(
            "normalize_double_precision", False
        ),
        learned_rewards=cfg.algorithm.learned_rewards,
        num_elites=cfg.overrides.get("num_elites", None),
    )
    if model_dir:
        dynamics_model.load(model_dir)

    return dynamics_model
