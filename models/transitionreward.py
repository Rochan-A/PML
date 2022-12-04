# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math

from mbrl.models.model import Ensemble, Model
import hydra
from .base import Backbone, ContextEncoder


class TransitionRewardModel(Model):
    """
    Model inputs/outputs will be consistent with

        [pred_obs_{t+1}, pred_rewards_{t+1} (optional)] = model([obs_t, action_t, history_vec]).

    To use with :class:mbrl.models.ModelEnv`, the wrapped model must define methods
    ``reset_1d`` and ``sample_1d``.

    Args:
        model (:class:`mbrl.model.Model`): the model to wrap.
        target_is_delta (bool): if ``True``, the predicted observations will represent
            the difference respect to the input observations.
            That is, ignoring rewards, pred_obs_{t + 1} = obs_t + model([obs_t, act_t]).
            Defaults to ``True``. Can be deactivated per dimension using ``no_delta_list``.
        normalize (bool): if true, the wrapper will create a normalizer for model inputs,
            which will be used every time the model is called using the methods in this
            class. Assumes the given base model has an attributed ``in_size``.
            To update the normalizer statistics, the user needs to call
            :meth:`update_normalizer` before using the model. Defaults to ``False``.
        normalize_double_precision (bool): if ``True``, the normalizer will work with
            double precision.
        learned_rewards (bool): if ``True``, the wrapper considers the last output of the model
            to correspond to rewards predictions, and will use it to construct training
            targets for the model and when returning model predictions. Defaults to ``True``.
        num_elites (int, optional): if provided, only the best ``num_elites`` models according
            to validation score are used when calling :meth:`predict`. Defaults to
            ``None`` which means that all models will always be included in the elite set.
    """

    def __init__(
        self,
        model: Model,
        context_cfg: dict,
        backbone_cfg: dict,
        use_context = False,
        target_is_delta: bool = True,
        normalize: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        num_elites: Optional[int] = None,
    ):
        super().__init__(model.device)

        self.use_context = use_context
        self.model = model
        self.context_enc = ContextEncoder(**context_cfg).to(self.model.device)
        self.backbone_enc = Backbone(**backbone_cfg).to(self.model.device)

        self.input_normalizer: Optional[mbrl.util.math.Normalizer] = None
        if normalize:

            if use_context:
                in_size = self.backbone_enc.in_dim + self.context_enc.in_dim
            else:
                in_size = self.model.in_size

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

    def _get_model_input(
        self,
        obs,
        action,
        context=None
    ) -> torch.Tensor:
        obs = model_util.to_tensor(obs).to(self.device)
        action = model_util.to_tensor(action).to(self.device)
        d = [obs, action]
        if context:
            context = model_util.to_tensor(context).to(self.device)
            d = [context,] + d
        model_in = torch.cat(d, dim=obs.ndim - 1)
        if self.input_normalizer:
            # Normalizer lives on device
            model_in = self.input_normalizer.normalize(model_in).float().to(self.device)
        return model_in

    def _process_batch(
        self, batch: mbrl.types.TransitionBatch, _as_float: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bb = batch.astuple()
        if len(bb) == 5:
            obs, action, next_obs, reward, _ = bb
            context = None
        else:
            context, obs, action, next_obs, reward, _ = bb
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

    # I think we can ignore this for now, not used anywhere I know
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
        # VERY IMPORTANT
        # split model_in -> [context, state, action]
        loss = 0
        if self.use_context:
            context, sa_t = model_in[:self.context_enc.in_dim, :], model_in[self.context_enc.in_dim:, :]
            c_embb = self.context_enc.forward(context)
            # loss += TODO: loss for context_enc
            b_embb = self.backbone_enc.forward(sa_t)
            model_in = torch.cat([c_embb, b_embb], dim=0)
        loss += self.model.loss(model_in, target=target)
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
        if self.use_context:
            # TODO:
            a = 0
        return self.model.update(model_in, optimizer, target=target)

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
                # TODO:
                a = 0
            return self.model.eval_score(model_in, target=target)

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
                # TODO:
                a = 0
            output = self.model.forward(model_in)
        return output, target

    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
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
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (random number generator): a rng to use for sampling.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        obs = model_util.to_tensor(model_state["obs"]).to(self.device)
        if self.use_context:
            # TODO:
            a = 0
        model_in = self._get_model_input(model_state["obs"], act)
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method sample_1d"
            )
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
        self.model.save(save_dir)
        self.context_enc.save(save_dir)
        self.backbone_enc.save(save_dir)
        if self.input_normalizer:
            self.input_normalizer.save(save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        self.model.load(load_dir)
        self.context_enc.load(load_dir)
        self.backbone_enc.load(load_dir)
        if self.input_normalizer:
            self.input_normalizer.load(load_dir)

    def set_elite(self, elite_indices: Sequence[int]):
        self.model.set_elite(elite_indices)

    def __len__(self):
        return len(self.model)

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        if isinstance(self.model, Ensemble):
            self.model.set_propagation_method(propagation_method)


def create_model(
    cfg,
    context_cfg,
    backbone_cfg,
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
        model,
        context_cfg,
        backbone_cfg,
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


if __name__=='__main__':
    import omegaconf

    trial_length = 200
    num_trials = 10
    ensemble_size = 5

    context_cfg = {
        'state_sz': 6,
        'action_sz': 1,
        'hidden_dim': 128,
        'hidden_layers': 1,
        'out_dim': 32,
        'history_size': 16
    }

    backbone_cfg = {
        'state_sz': 6,
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

    dynamics_model = create_model(cfg, context_cfg, backbone_cfg)
    print(dynamics_model.forward(torch.zeros((10, 64)), propagation_indices=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))

    model = mbrl.models.gaussian_mlp.GaussianMLP(64, 2, torch.device('cpu'), ensemble_size=5, propagation_method="random_model")
    print(model.forward(torch.zeros((10, 64))))