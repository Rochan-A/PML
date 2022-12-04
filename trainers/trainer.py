import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf

import mbrl.util.common as common_util
import mbrl.models as models

from models.transitionreward import *
from models.dynamics_model import *
from .replay_buffer import create_replay_buffer
from .agent import create_agent


def rollout_agent_trajectories(
    env: gym.Env,
    steps_or_trials_to_collect: int,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    trial_length: Optional[int] = None,
    replay_buffer = None,
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
            trials (full trials in each dataset); otherwise, it's done across steps.

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

    step = 0
    trial = 0
    total_rewards: List[float] = []
    while True:
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(obs, **agent_kwargs)
            next_obs, reward, done, info = env.step(action)

            if replay_buffer is not None:
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
        reward_fn,
        term_fn,
        config,
        args,
        writer,
        no_test_flag=False,
        only_test_flag=False,
    ):

        self.trial_length = config.trail_length
        self.num_trials = config.num_trials
        self.ensemble_size = config.head.ensemble_size

        generator = torch.Generator(device=config["device"])
        generator.manual_seed(args.seed)

        # Everything with "???" indicates an option with a missing value.
        # Our utility functions will fill in these details using the
        # environment information
        cfg_dict = {
            # dynamics model configuration
            "dynamics_model": {
                "model": {
                    "_target_": "mbrl.models.GaussianMLP",
                    "device": str(config["device"]),
                    "num_layers": config.head.hidden_layers,
                    "ensemble_size": self.ensemble_size,
                    "hid_size": config.head.hidden_dim,
                    "in_size": env.observation_space.shape[0]
                    + env.action_space.shape[0],
                    "out_size": "???",
                    "deterministic": False,
                    "propagation_method": "fixed_model",
                    # can also configure activation function for GaussianMLP
                    "activation_fn_cfg": {
                        "_target_": "torch.nn.LeakyReLU",
                        "negative_slope": 0.01,
                    },
                }
            },
            # options for training the dynamics model
            "algorithm": {
                "learned_rewards": True,
                "target_is_delta": True,
                "normalize": config.normalize_flag,
            },
            # these are experiment specific options
            "overrides": {
                "trial_length": self.trial_length,
                "num_steps": self.num_trials * self.trial_length,
                "model_batch_size": config.dynamics.batch_size,
                "validation_ratio": config.dynamics.validation_ratio,
            },
        }
        self.cfg = omegaconf.OmegaConf.create(cfg_dict)

        # Contruct context, backbone encoder configs
        self.context_cfg = config.context
        self.context_cfg["state_sz"] = env.observation_space.shape[0]
        self.context_cfg["action_sz"] = env.action_space.shape[0]
        self.backbone_cfg = config.backbone
        self.backbone_cfg["state_sz"] = env.observation_space.shape[0]
        self.backbone_cfg["action_sz"] = env.action_space.shape[0]

        # Create a dynamics model for this environment
        self.dynamics_model = create_model(
            self.cfg, self.context_cfg, self.backbone_cfg
        )

        # Create custom gym-like environment to encapsulate the model TODO:
        self.model_env = ModelEnv(
            env, self.dynamics_model, term_fn, reward_fn, generator=generator
        )

        # Create custom replay buffer
        self.replay_buffer = create_replay_buffer(
            self.cfg,
            env.observation_space.shape,
            env.action_space.shape,
            rng=np.random.default_rng(args.seed),
        )

        # Initialize buffer with some trajectories
        _ = rollout_agent_trajectories(
            env,
            self.trial_length,  # initial exploration steps
            mbrl.planning.RandomAgent(env),
            {},  # keyword arguments to pass to agent.act()
            replay_buffer=self.replay_buffer,
            trial_length=self.trial_length,
        )
        print("# samples stored", self.replay_buffer.num_stored)

        agent_cfg = omegaconf.OmegaConf.create(
            {
                # this class evaluates many trajectories and picks the best one
                "_target_": "TrajectoryOptimizerAgent",
                "planning_horizon": config.agent.horizon,
                "replan_freq": config.agent.replan_freq,
                "verbose": False,
                "action_lb": "???",
                "action_ub": "???",
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
        )

        # create agent
        self.agent = create_agent(agent_cfg, self.model_env, config)

        # Create a trainer for the model
        self.model_trainer = models.ModelTrainer(
            self.dynamics_model, optim_lr=config.dynamics.learning_rate, weight_decay=5e-5
        )

    def run(self, env):
        train_losses = []
        val_scores = []

        def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
            train_losses.append(tr_loss)
            val_scores.append(
                val_score.mean().item()
            )  # this returns val score per ensemble model

        # Main PETS loop
        all_rewards = [0]
        for trial in range(self.num_trials):
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
                        batch_size=self.cfg.overrides.model_batch_size,
                        val_ratio=self.cfg.overrides.validation_ratio,
                        ensemble_size=self.ensemble_size,
                        shuffle_each_epoch=True,
                        bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                    )

                    self.model_trainer.train(
                        dataset_train,
                        dataset_val=dataset_val,
                        num_epochs=50,
                        patience=50,
                        callback=train_callback,
                    )

                # --- Doing env step using the agent and adding to model dataset ---
                # next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
                #     env, obs, agent, {}, replay_buffer)

                action = self.agent.act(obs, **{})
                next_obs, reward, done, _ = env.step(action)
                self.replay_buffer.add(obs, action, next_obs, reward, done)

                obs = next_obs
                total_reward += reward
                steps_trial += 1

                if steps_trial == self.trial_length:
                    break

            all_rewards.append(total_reward)
        return train_losses, val_scores, all_rewards


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
        plt.ylabel(ylabel)
        plt.plot(data, "bs-")
        plt.savefig(path)
