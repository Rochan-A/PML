from copy import deepcopy
from pprint import pprint
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

import gym

from mbrl.util.logger import Logger
import mbrl.models as models
import omegaconf
import torch
import numpy as np
import torch.distributions
from mbrl.planning import Agent, RandomAgent, create_trajectory_optim_agent_for_model

import mbrl.util.common as common_util

from tqdm import tqdm
from envs import ContextEnv, DummyContextEnv
from replay_buffer import (
    ReplayBuffer,
    RollingHistory,
)


def rollout_agent_trajectories(
    env_fam: Union[ContextEnv, DummyContextEnv],
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    steps_or_trials_to_collect: int,
    agent: Agent,
    agent_kwargs: Dict,
    ctx_hist_len: Optional[int] = None,
    trial_length: Optional[int] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    collect_full_trajectories: bool = False,
) -> List[float]:
    """Rollout agent trajectories in the given environment.

    Rollouts trajectories in the environment using actions produced by the given agent.
    Optionally, it stores the saved data into a replay buffer.

    Args:
        env_fam (ContextEnv | DummyContextEnv): Environment to rollout the agent in.
        obs_shape (tuple): shape of the observation space.
        act_shape (tuple): shape of the action space.
        steps_or_trials_to_collect (int): how many steps of the environment to collect. If
            ``collect_trajectories=True``, it indicates the number of trials instead.
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        ctx_hist_len (int, optional): if not ``None``, indicates use a rolling history buffer
        trial_length (int, optional): a maximum length for trials (env will be reset regularly
            after this many number of steps). Defaults to ``None``, in which case trials
            will end when the environment returns ``done=True``.
        replay_buffer (:class:`mbrl.util.ReplayBuffer`, optional):
            a replay buffer to store data to use for training.
        collect_full_trajectories (bool): if ``True``, indicates that replay buffers should
            collect full trajectories. This only affects the split between training and
            validation buffers. If ``collect_trajectories=True``, the split is done over
            trials.reset(train=True) (full trials in each dataset); otherwise, it's done across steps.

    Returns:
        (list(float)): Total rewards obtained at each complete trial.
    """
    if (
        replay_buffer is not None
        and replay_buffer.stores_trajectories
        and not collect_full_trajectories
    ):
        # Might be better as a warning but it's possible that users will miss it.
        raise RuntimeError(
            "Replay buffer is tracking trajectory information but "
            "collect_trajectories is set to False, which will result in "
            "corrupted trajectory data."
        )

    step, trial = 0, 0
    total_rewards: List[float] = []

    if ctx_hist_len:
        rhc = RollingHistory(
            T=ctx_hist_len,
            obs_shape=obs_shape,
            act_shape=act_shape,
        )

    while True:
        env, _ = env_fam.reset()

        obs = env.reset()
        if ctx_hist_len:
            rhc.reset()
            rhc.append(obs, None)
        agent.reset()
        done = False
        total_reward = 0.0
        while not done:
            if ctx_hist_len:
                action = agent.act(obs, ctx=deepcopy(rhc.store), **agent_kwargs)
                rhc.append(obs, action)
            else:
                action = agent.act(obs, **agent_kwargs)

            next_obs, reward, done, info = env.step(action)

            if replay_buffer is not None:
                if ctx_hist_len:
                    replay_buffer.add(
                        obs, action, next_obs, reward, done, deepcopy(rhc.store)
                    )
                else:
                    replay_buffer.add(obs, action, next_obs, reward, done)

            obs = next_obs
            total_reward += reward
            step += 1

            if not collect_full_trajectories and step == steps_or_trials_to_collect:
                total_rewards.append(total_reward)
                return total_rewards

            if trial_length and step % trial_length == 0:
                if collect_full_trajectories and not done and replay_buffer is not None:
                    replay_buffer.close_trajectory()
                break
        trial += 1
        total_rewards.append(total_reward)
        if collect_full_trajectories and trial == steps_or_trials_to_collect:
            break

    return total_rewards


class PetsBase:
    """Abstract class for training PETS model.

    Args:
        dummy_env (gym.Env): A dummy environment to use for observation_space and action_space.
        env_fam (ContextEnv): Environment to rollout the agent in.
        reward_fn (callable): A function that takes in a state and action and returns a reward.
        term_fn (callable): A function that takes in a state and action and returns a boolean indicating if the episode is over.
        config (dict): A dictionary containing the configuration parameters.
        rng (np.random.Generator): A random number generator.
        save_path (str): Path to save the results to.
    """

    def __init__(
        self,
        *,
        dummy_env: gym.Env,
        env_fam: ContextEnv,
        reward_fn: Callable,
        term_fn: Callable,
        config: dict[str, Any],
        rng: np.random.Generator,
        save_path: str,
    ) -> None:

        self.trial_len = config.get("trial_len", 200)
        self.num_trials = config.get("num_trials", 20)
        self.init_trials = config.get("init_trials", 20)
        self.save_path = save_path
        self.obs_shape, self.act_shape = (
            dummy_env.observation_space.shape,
            dummy_env.action_space.shape,
        )
        self.env_fam = env_fam
        self.rng = rng
        self.logger = Logger(self.save_path)
        self.device = config.device
        self.generator = torch.Generator(device=self.device)

        self.dynamics_model_args = config.dynamics_model_args
        self.dynamics_train_args = config.dynamics_train_args
        self.agent_args = config.agent_args
        self.cem_optim_args = config.cem_optim_args
        self.ensemble_size = self.dynamics_model_args.get("ensemble_size", 1)

        # Everything with "???" indicates an option with a missing value.
        # Our utility functions will fill in these details using the
        # environment information
        cfg_dict = {
            # dynamics model configuration
            "dynamics_model": {
                "model": {
                    "_target_": "mbrl.models.GaussianMLP",
                    "device": str(config["device"]),
                    "num_layers": self.dynamics_model_args.get("num_layers", 2),
                    "ensemble_size": self.ensemble_size,
                    "hid_size": self.dynamics_model_args.get("hidden_dim", 200),
                    "in_size": "???",
                    "out_size": "???",
                    "deterministic": self.dynamics_model_args.get(
                        "deterministic", False
                    ),
                    "propagation_method": self.dynamics_model_args.get(
                        "prop_method", "???"
                    ),
                    # can also configure activation function for GaussianMLP
                    "activation_fn_cfg": {
                        "_target_": self.dynamics_model_args.get(
                            "actv", "torch.nn.ReLU"
                        ),
                        "negative_slope": 0.01,
                    },
                }
            },
            # options for training the dynamics model
            "algorithm": {
                "learned_rewards": True if reward_fn is None else False,
                "target_is_delta": config.target_is_delta,
                "normalize": config.normalize_flag,
                "dataset_size": config.replay_buffer_sz,
            },
            # these are experiment specific options
            "overrides": {
                "trial_length": self.trial_len,
                "model_batch_size": self.dynamics_train_args.batch_size,
                "validation_ratio": self.dynamics_train_args.validation_ratio,
            },
        }
        print("1-D Transition Dynamics Model Configuration:")
        pprint(cfg_dict)
        cfg = omegaconf.OmegaConf.create(cfg_dict)

        # Create a 1-D dynamics model for this environment
        self.dynamics_model = common_util.create_one_dim_tr_model(
            cfg=cfg, obs_shape=self.obs_shape, act_shape=self.act_shape
        )

        # Create a gym-like environment to encapsulate the model
        self.model_env = models.ModelEnv(
            env=dummy_env,
            model=self.dynamics_model,
            termination_fn=term_fn,
            reward_fn=reward_fn,
            generator=self.generator,
        )

        agent_cfg = {
            # this class evaluates many trajectories and picks the best one
            "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
            "planning_horizon": self.agent_args.horizon,
            "replan_freq": self.agent_args.replan_freq,
            "verbose": False,
            "action_lb": [-1.0],
            "action_ub": [1.0],
            # this is the optimizer to generate and choose a trajectory
            "optimizer_cfg": {
                "_target_": "mbrl.planning.CEMOptimizer",
                "num_iterations": self.cem_optim_args.num_iters,
                "elite_ratio": self.cem_optim_args.elite_ratio,
                "population_size": self.cem_optim_args.popsize,
                "alpha": self.cem_optim_args.alpha,
                "device": str(config["device"]),
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": True,
            },
        }
        print("Agent + Optimizer Configuration:")
        pprint(agent_cfg)
        agent_cfg = omegaconf.OmegaConf.create(agent_cfg)

        # Create a trajectory optimizer agent
        self.agent = create_trajectory_optim_agent_for_model(
            model_env=self.model_env,
            agent_cfg=agent_cfg,
            num_particles=self.agent_args.max_particles,
        )

        # Keep the configuration for later use
        self.cfg = cfg

    def run(self, **kwargs) -> None:
        """Runs the training loop.

        Args:
            kwargs: Any keyword arguments to pass to the training loop.
        """
        raise NotImplementedError


class PETSTrainer(PetsBase):
    """Trainer for the Probabilistic Ensembles with Trajectory Sampling (PETS) algorithm."""

    def __init__(
        self,
        *,
        dummy_env: gym.Env,
        env_fam: ContextEnv,
        reward_fn: Callable,
        term_fn: Callable,
        config: dict[str, Any],
        rng: np.random.Generator,
        save_path: str,
    ) -> None:
        super().__init__(
            dummy_env=dummy_env,
            env_fam=env_fam,
            reward_fn=reward_fn,
            term_fn=term_fn,
            config=config,
            rng=rng,
            save_path=save_path,
        )

        # Create a replay buffer for the model
        self.replay_buffer = common_util.create_replay_buffer(
            cfg=self.cfg, obs_shape=self.obs_shape, act_shape=self.act_shape, rng=rng
        )

        # Initialize buffer with some trajectories
        _ = rollout_agent_trajectories(
            env_fam=env_fam,
            obs_shape=self.obs_shape,
            act_shape=self.act_shape,
            steps_or_trials_to_collect=self.init_trials,
            agent=RandomAgent(dummy_env),
            agent_kwargs={},
            ctx_hist_len=None,
            trial_length=self.trial_len,
            replay_buffer=self.replay_buffer,
            collect_full_trajectories=False,
        )
        print("# samples stored", self.replay_buffer.num_stored)

        # Create a trainer for the model
        self.model_trainer = models.ModelTrainer(
            model=self.dynamics_model,
            optim_lr=self.dynamics_train_args.lr,
            weight_decay=5e-5,
        )

    def run(self, *, writer) -> dict[str, Any]:
        """Run PETS

        Args:
            writer (SummaryWriter): Tensorboard writer

        Returns:
            dict[str, Any]: Dictionary of model & metrics
        """
        train_losses = []
        val_scores = []

        def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
            train_losses.append(tr_loss)
            val_scores.append(
                val_score.mean().item()
            )  # this returns val score per ensemble model
            writer.add_scalar("loss/train", tr_loss, _epoch)
            writer.add_scalar("score/val_score", val_score.mean().item(), _epoch)

        # Main PETS loop
        all_rewards, all_contexts = [], []

        pb = tqdm(range(self.num_trials), desc="Running PETS Trainer")
        for trial, _ in enumerate(pb):
            env, ctx_vals = self.env_fam.reset()
            all_contexts.append(ctx_vals)

            obs = env.reset()
            self.agent.reset()

            done = False
            total_reward = 0.0
            steps_trial = 0

            while not done:
                # --------------- Model Training -----------------
                if steps_trial == 0:
                    self.dynamics_model.update_normalizer(
                        self.replay_buffer.get_all()
                    )  # update normalizer stats

                    dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                        self.replay_buffer,
                        batch_size=self.dynamics_train_args.batch_size,
                        val_ratio=self.dynamics_train_args.validation_ratio,
                        ensemble_size=self.ensemble_size,
                        shuffle_each_epoch=True,
                        bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                    )

                    self.model_trainer.train(
                        dataset_train=dataset_train,
                        dataset_val=dataset_val,
                        num_epochs=self.dynamics_train_args.epochs_per_step,
                        patience=self.dynamics_train_args.patience,
                        callback=train_callback,
                    )

                next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
                    env, obs, self.agent, {}, self.replay_buffer
                )

                obs = next_obs
                total_reward += reward
                steps_trial += 1

                if steps_trial == self.trial_len:
                    break
            writer.add_scalar("reward/train", total_reward, trial)
            all_rewards.append(total_reward)

            print()
            ctx_str = " | " if ctx_vals is None else f" | Context: {ctx_vals} |"
            pb.set_description(
                f"Episode: {trial}{ctx_str}Cumulative reward: {total_reward}"
            )

        self.dynamics_model.save(self.save_path)

        return {
            "train_losses": train_losses,
            "val_scores": val_scores,
            "rewards": all_rewards,
            "model": self.model_env,
            "all_contexts": all_contexts,
        }


class PETSTester(PetsBase):
    """Tester for the Probabilistic Ensembles with Trajectory Sampling (PETS) algorithm."""

    def __init__(
        self,
        *,
        dummy_env: gym.Env,
        env_fam: ContextEnv,
        reward_fn: Callable,
        term_fn: Callable,
        config: dict[str, Any],
        rng: np.random.Generator,
        save_path: str,
        load_path: str,
        model=None,
    ) -> None:
        super().__init__(
            dummy_env=dummy_env,
            env_fam=env_fam,
            reward_fn=reward_fn,
            term_fn=term_fn,
            config=config,
            rng=rng,
            save_path=save_path,
        )

        # Load model
        if model is None:
            assert load_path is not None, "Must provide load path if model is None"
            self.dynamics_model.load(load_path)
        else:
            assert isinstance(model, models.ModelEnv), "Model must be of type ModelEnv"
            self.dynamics_model = model

    def run(self, *, num_trials: int) -> dict[str, Any]:
        """Test PETS

        Args:
            num_trials (int): Number of trials to run for each context

        Returns:
            dict[str, Any]: Dictionary of model & metrics
        """
        all_rewards = []
        all_contexts = []

        pb = tqdm(range(len(self.env_fam)), desc="Running PETS Tester")
        # len(env_fam) returns the number of MDPs to test over
        # iterate over all possible MDP permutations
        for trial, _ in enumerate(pb):

            env, ctx_vals = self.env_fam.reset(idx=trial)
            all_contexts.append(ctx_vals)

            rewards = np.zeros(num_trials)
            for trial in range(num_trials):

                obs = env.reset()
                self.agent.reset()

                done = False
                total_reward = 0.0
                steps_trial = 0

                while not done:

                    action = self.agent.act(obs, **{})
                    next_obs, reward, done, _ = env.step(action)

                    obs = next_obs
                    total_reward += reward
                    steps_trial += 1

                    if steps_trial == self.trial_len:
                        break
                rewards[trial] = total_reward

            all_rewards.append(rewards)

            ctx_str = " | " if ctx_vals is None else f" | Context: {ctx_vals} |"
            pb.set_description(
                f"Episode: {trial}{ctx_str}Mean Cumulative reward: {rewards.mean()}"
            )

        return {"rewards": all_rewards, "all_contexts": all_contexts}
