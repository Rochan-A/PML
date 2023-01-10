import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf
from pprint import pprint
from os.path import join
from copy import deepcopy
import mbrl.models as models

from models.transitionreward import create_model
from models.dynamics_model import ModelEnv
from .replay_buffer import create_replay_buffer

import time
from typing import Callable, List, Optional, Sequence, cast, Dict

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions

import mbrl.models
import mbrl.types
import mbrl.util.math

from mbrl.planning.core import Agent
from mbrl.planning.trajectory_opt import TrajectoryOptimizer
from .replay_buffer import get_basic_buffer_iterators, create_replay_buffer, ReplayBuffer


class RollingHistoryContext:
    def __init__(self, K, state_sz, action_sz) -> None:
        self.K = K

        self.state_sz = state_sz
        self.action_sz = action_sz
        self.default_sz = (state_sz + action_sz) * K
        self.store = None
        self.prev_st = None

    def append(self, state, action=None):
        if self.prev_st is None:
            self.prev_st = state
            n_state = state
        else:
            n_state = state# - self.prev_st
            self.prev_st = state

        if action is None:
            action = np.random.normal(scale=1e-3, size=self.action_sz)

        k = np.concatenate([n_state, action], axis=0)
        if self.store is None:
            self.store = np.tile(deepcopy(k), self.K)
        else:
            self.store = np.roll(self.store, -k.shape[0])
            self.store[-k.shape[0] :] = deepcopy(k)

    def reset(self):
        self.store = None
        self.prev_st = None


class TrajectoryOptimizerAgent(Agent):
    """Agent that performs trajectory optimization on a given objective function for each action.

    This class uses an internal :class:`TrajectoryOptimizer` object to generate
    sequence of actions, given a user-defined trajectory optimization function.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the base optimizer to pass to
            the trajectory optimizer.
        action_lb (sequence of floats): the lower bound of the action space.
        action_ub (sequence of floats): the upper bound of the action space.
        planning_horizon (int): the length of action sequences to evaluate. Defaults to 1.
        replan_freq (int): the frequency of re-planning. The agent will keep a cache of the
            generated sequences an use it for ``replan_freq`` number of :meth:`act` calls.
            Defaults to 1.
        verbose (bool): if ``True``, prints the planning time on the console.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using the initial
            solution. Defaults to ``True``.

    Note:
        After constructing an agent of this type, the user must call
        :meth:`set_trajectory_eval_fn`. This is not passed to the constructor so that the agent can
        be automatically instantiated with Hydra (which in turn makes it easy to replace this
        agent with an agent of another type via config-only changes).
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        planning_horizon: int = 1,
        replan_freq: int = 1,
        verbose: bool = False,
        keep_last_solution: bool = True,
    ):
        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg,
            np.array(action_lb),
            np.array(action_ub),
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
            keep_last_solution=keep_last_solution,
        )
        self.optimizer_args = {
            "optimizer_cfg": optimizer_cfg,
            "action_lb": np.array(action_lb),
            "action_ub": np.array(action_ub),
        }
        self.trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.actions_to_use: List[np.ndarray] = []
        self.replan_freq = replan_freq
        self.verbose = verbose

    def set_trajectory_eval_fn(
        self, trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        """Sets the trajectory evaluation function.

        Args:
            trajectory_eval_fn (callable): a trajectory evaluation function, as described in
                :class:`TrajectoryOptimizer`.
        """
        self.trajectory_eval_fn = trajectory_eval_fn

    def reset(self, planning_horizon: Optional[int] = None):
        """Resets the underlying trajectory optimizer."""
        if planning_horizon:
            self.optimizer = TrajectoryOptimizer(
                cast(omegaconf.DictConfig, self.optimizer_args["optimizer_cfg"]),
                cast(np.ndarray, self.optimizer_args["action_lb"]),
                cast(np.ndarray, self.optimizer_args["action_ub"]),
                planning_horizon=planning_horizon,
                replan_freq=self.replan_freq,
            )

        self.optimizer.reset()

    def act(
        self,
        obs: np.ndarray,
        context=None,
        optimizer_callback: Optional[Callable] = None,
        **_kwargs,
    ) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.

        Args:
            obs (np.ndarray): the observation for which the action is needed.
            optimizer_callback (callable, optional): a callback function
                to pass to the optimizer.
            context (np.ndarray, Optional): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )
        plan_time = 0.0
        if not self.actions_to_use:  # re-plan is necessary

            def trajectory_eval_fn(action_sequences):
                return self.trajectory_eval_fn(obs, action_sequences, context)

            start_time = time.time()
            plan = self.optimizer.optimize(
                trajectory_eval_fn, callback=optimizer_callback
            )
            plan_time = time.time() - start_time

            self.actions_to_use.extend([a for a in plan[: self.replan_freq]])
        action = self.actions_to_use.pop(0)

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action

    def plan(self, obs: np.ndarray, context=None, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Returns s sequence of length self.planning_horizon.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )

        def trajectory_eval_fn(action_sequences):
            return self.trajectory_eval_fn(obs, action_sequences, context)

        plan = self.optimizer.optimize(trajectory_eval_fn)
        return plan


def create_agent(agent_cfg, model_env, config):
    mbrl.planning.complete_agent_cfg(model_env, agent_cfg)
    agent = hydra.utils.instantiate(agent_cfg)

    # Get details of the eval function we want to use
    eval_cfg = config.agent.eval

    EVAL_FN = {
        "default": model_env.evaluate_action_sequences,
        "kl": model_env.evaluate_action_sequences_kl,
        "greedy": model_env.evaluate_action_sequences_greedy,
        "combined": model_env.evaluate_action_sequences_combined,
    }

    if eval_cfg.method == "combine":
        raise NotImplementedError

    def trajectory_eval_fn(initial_state, action_sequences, initial_context=None):
        return EVAL_FN[eval_cfg.method](
            action_sequences,
            initial_state=initial_state,
            num_particles=config.agent.max_particles,
            initial_context=initial_context,
        )

    agent.set_trajectory_eval_fn(trajectory_eval_fn)
    return agent


def rollout_agent_trajectories(
    env_fam,
    steps_or_trials_to_collect: int,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    context_len: int = None,
    trial_length: Optional[int] = None,
    replay_buffer: ReplayBuffer = None,
    collect_full_trajectories: bool = False,
) -> List[float]:
    """Rollout agent trajectories in the given environment.

    Rollouts trajectories in the environment using actions produced by the given agent.
    Optionally, it stores the saved data into a replay buffer.

    Args:
        env (gym.Env): the environment to step.
        steps_or_trials_to_collect (int): how many steps of the environment to collect. If
            ``collect_trajectories=True``, it indicates the number of trials instead.
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
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

    env, _ = env_fam.reset()
    step = 0
    trial = 0
    total_rewards: List[float] = []
    if context_len:
        rhc = RollingHistoryContext(
            context_len, env.observation_space.shape[0], env.action_space.shape[0]
        )
    while True:
        env, _ = env_fam.reset()

        obs = env.reset()
        if context_len:
            rhc.reset()
            rhc.append(obs, None)
        agent.reset()
        done = False
        total_reward = 0.0
        while not done:
            if context_len:
                action = agent.act(obs, deepcopy(rhc.store), **agent_kwargs)
                rhc.append(obs, action)
            else:
                action = agent.act(obs, **agent_kwargs)

            next_obs, reward, done, info = env.step(action)

            if replay_buffer is not None:
                if context_len:
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


class Trainer(object):
    def __init__(
        self,
        env,
        env_fam,
        reward_fn,
        term_fn,
        config,
        args,
        writer,
        no_test_flag,
        rng,
        save_path,
    ):

        self.trial_length = config.trail_length
        self.num_trials = config.num_trials
        self.ensemble_size = config.transitionreward.ensemble_size
        self.context_len = (
            None if config.context.no_context else config.context.history_size
        )
        self.writer = writer
        self.no_test_flag = no_test_flag
        self.epochs_per_step = config.dynamics.epochs_per_step
        self.patience = config.dynamics.patience

        generator = torch.Generator(device=config["device"])
        generator.manual_seed(args.seed)

        # Everything with "???" indicates an option with a missing value.
        # Our utility functions will fill in these details using the
        # environment information
        in_sz = (
            config.context.out_dim + config.stateaction.out_dim
            if self.context_len is not None
            else config.stateaction.out_dim
        )

        cfg_dict = {
            # dynamics model configuration
            "dynamics_model": {
                "model": {
                    "_target_": "mbrl.models.GaussianMLP",
                    "device": str(config["device"]),
                    "num_layers": config.transitionreward.hidden_layers,
                    "ensemble_size": self.ensemble_size,
                    "hid_size": config.transitionreward.hidden_dim,
                    "in_size": in_sz,
                    "out_size": "???",
                    "deterministic": False,
                    "propagation_method": config.transitionreward.prop_method,
                    # can also configure activation function for GaussianMLP
                    "activation_fn_cfg": {
                        "_target_": config.transitionreward.actv,
                        "negative_slope": 0.01,
                    },
                }
            },
            # options for training the dynamics model
            "algorithm": {
                "learned_rewards": True if reward_fn is None else False,
                "target_is_delta": True,
                "normalize": config.normalize_flag,
                "dataset_size": config.replay_buffer_sz,
            },
            # these are experiment specific options
            "overrides": {
                "trial_length": self.trial_length,
                "model_batch_size": config.dynamics.batch_size,
                "validation_ratio": config.dynamics.validation_ratio,
            },
        }
        print("Model Env cfg")
        pprint(cfg_dict)
        self.cfg = omegaconf.OmegaConf.create(cfg_dict)

        # Contruct context, stateaction encoder configs
        self.context_cfg = config.context
        self.context_cfg["state_sz"] = env.observation_space.shape[0]
        self.context_cfg["action_sz"] = env.action_space.shape[0]
        self.stateaction_cfg = config.stateaction
        self.stateaction_cfg["state_sz"] = env.observation_space.shape[0]
        self.stateaction_cfg["action_sz"] = env.action_space.shape[0]

        # Create a dynamics model for this environment
        self.dynamics_model = create_model(
            self.cfg,
            self.context_cfg,
            self.stateaction_cfg,
            config.agent.eval,
            False if self.context_len is None else True,
        )

        # Create custom gym-like environment to encapsulate the model
        self.model_env = ModelEnv(
            env, self.dynamics_model, term_fn, reward_fn, generator=generator
        )

        # Create custom replay buffer
        self.replay_buffer = create_replay_buffer(
            self.cfg,
            env.observation_space.shape,
            env.action_space.shape,
            context_len=self.context_len,
            rng=rng,
        )

        # Initialize buffer with some trajectories
        _ = rollout_agent_trajectories(
            env_fam=env_fam,
            steps_or_trials_to_collect=config.init_trials,
            agent=mbrl.planning.RandomAgent(env),
            agent_kwargs={},  # keyword arguments to pass to agent.act(),
            context_len=self.context_len,
            replay_buffer=self.replay_buffer,
            trial_length=self.trial_length,
            collect_full_trajectories=True,
        )
        print("# samples stored", self.replay_buffer.num_stored)

        agent_cfg = {
            # this class evaluates many trajectories and picks the best one
            "_target_": "trainers.TrajectoryOptimizerAgent",
            "planning_horizon": config.agent.horizon,
            "replan_freq": config.agent.replan_freq,
            "verbose": False,
            "action_lb": [-1.0],
            "action_ub": [1.0],
            # this is the optimizer to generate and choose a trajectory
            "optimizer_cfg": {
                "_target_": "mbrl.planning.CEMOptimizer",
                "num_iterations": config.agent.num_iters,
                "elite_ratio": config.agent.elite_ratio,
                "population_size": config.agent.horizon,
                "alpha": config.agent.alpha,
                "device": str(config["device"]),
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": True,
            },
        }
        print("Agent Optimizer cfg")
        pprint(agent_cfg)
        agent_cfg = omegaconf.OmegaConf.create(agent_cfg)

        # create agent
        self.agent = create_agent(agent_cfg, self.model_env, config)

        logger = None
        if args.logger:
            from mbrl.util.logger import Logger
            logger = Logger(save_path)

        # Create a trainer for the model
        self.model_trainer = models.ModelTrainer(
            self.dynamics_model,
            optim_lr=config.dynamics.learning_rate,
            weight_decay=5e-5,
            logger=logger,
        )

    def run(self, env_fam, env, PATH):
        train_losses = []
        val_scores = []

        def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
            train_losses.append(tr_loss)
            val_scores.append(
                val_score.mean().item()
            )  # this returns val score per ensemble model
            self.writer.add_scalar("loss/train", tr_loss, _epoch)
            self.writer.add_scalar("score/val_score", val_score.mean().item(), _epoch)

        if self.context_len:
            rhc = RollingHistoryContext(
                K=self.context_len,
                state_sz=env.observation_space.shape[0],
                action_sz=env.action_space.shape[0],
            )

        # Main PETS loop
        all_rewards = [0]
        all_contexts = [None]
        for trial in range(self.num_trials):

            # Sample CMDP from distribution. If --mdp flag, then it returns the
            # same MDP.
            env, ctx_vals = env_fam.reset()
            all_contexts.append(ctx_vals)
            print(
                "trial: {}\t Context vector: {}".format(
                    trial, ctx_vals if ctx_vals is not None else "<fixed>"
                )
            )

            obs = env.reset()
            if self.context_len:
                rhc.reset()
                rhc.append(obs, None)

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

                    dataset_train, dataset_val = get_basic_buffer_iterators(
                        self.replay_buffer,
                        batch_size=self.cfg.overrides.model_batch_size,
                        val_ratio=self.cfg.overrides.validation_ratio,
                        ensemble_size=self.ensemble_size,
                        shuffle_each_epoch=True,
                        bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                    )

                    self.model_trainer.train(
                        dataset_train=dataset_train,
                        dataset_val=dataset_val,
                        num_epochs=self.epochs_per_step,
                        patience=self.patience,
                        callback=train_callback,
                    )

                # --- Doing env step using the agent and adding to model dataset ---
                if self.context_len:
                    action = self.agent.act(obs, deepcopy(rhc.store), **{})
                    rhc.append(obs, action)
                else:
                    action = self.agent.act(obs, **{})
                next_obs, reward, done, _ = env.step(action)
                if self.context_len:
                    self.replay_buffer.add(
                        obs, action, next_obs, reward, done, deepcopy(rhc.store)
                    )
                else:
                    self.replay_buffer.add(obs, action, next_obs, reward, done)

                obs = next_obs
                total_reward += reward
                steps_trial += 1

                if steps_trial == self.trial_length:
                    break
            self.writer.add_scalar("reward/train", total_reward, trial)
            all_rewards.append(total_reward)

        self.dynamics_model.save(PATH)

        # Make plots
        self.plot(
            [train_losses, val_scores],
            join(PATH, "scores_losses.png"),
            ["Epoch", "Epoch"],
            ["Training loss (avg. NLL)", "Validation score (avg. MSE)"],
        )
        self.plot_single(
            all_rewards, join(PATH, "rewards.png"), xlabel="Trial", ylabel="Reward"
        )

        if self.no_test_flag:
            return {
                "train_losses": train_losses,
                "val_scores": val_scores,
                "rewards": all_rewards,
                "all_contexts": all_contexts,
            }
        else:
            return {
                "train_losses": train_losses,
                "val_scores": val_scores,
                "rewards": all_rewards,
                "model": self.model_env,
                "all_contexts": all_contexts,
            }

    def plot(self, data, path, xlabels, ylabels):
        _, ax = plt.subplots(len(data), 1)
        for (i, dat, xlabel, ylabel) in zip(range(len(data)), data, xlabels, ylabels):
            ax[i].plot(dat)
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)
        plt.savefig(path)

    def plot_single(self, data, path, xlabel, ylabel):
        plt.figure(dpi=200)
        plt.xlabel(xlabel)
        plt.yticks(np.arange(0, 200, 20))
        plt.ylim([0, 200])
        plt.ylabel(ylabel)
        plt.plot(data, "bs-")
        plt.savefig(path)
