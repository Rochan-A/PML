import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf

import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.models as models

from models.transitionreward import *
from models.dynamics_model import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Trainer(object):
    def __init__(
        self,
        env,
        config,
        writer,
        no_test_flag=False,
        only_test_flag=False,
    ):

        self.trial_length = config.trail_length
        self.num_trials = config.num_trials
        self.ensemble_size = config.head.ensemble_size

        # Everything with "???" indicates an option with a missing value.
        # Our utility functions will fill in these details using the
        # environment information
        cfg_dict = {
            # dynamics model configuration
            "dynamics_model": {
                "model":
                {
                    "_target_": "mbrl.models.GaussianMLP",
                    "device": device,
                    "num_layers": config.head.hidden_layers,
                    "ensemble_size": self.ensemble_size,
                    "hid_size": config.head.hidden_dim,
                    "in_size": env.observation_space.shape[0]+env.action_space.shape[0],
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
                "learned_rewards": True,
                "target_is_delta": True,
                "normalize": config.normalize_flag,
            },
            # these are experiment specific options
            "overrides": {
                "trial_length": self.trial_length,
                "num_steps": self.num_trials * self.trial_length,
                "model_batch_size": 32,
                "validation_ratio": 0.05
            }
        }
        self.cfg = omegaconf.OmegaConf.create(cfg_dict)

        # Context, backbone encoder configs
        self.context_cfg = config.context
        self.context_cfg['state_sz'] = env.observation_space.shape[0]
        self.context_cfg['action_sz'] = env.action_space.shape[0]
        self.backbone_cfg = config.backbone
        self.backbone_cfg['state_sz'] = env.observation_space.shape[0]
        self.backbone_cfg['action_sz'] = env.action_space.shape[0]

        # Create a dynamics model for this environment
        self.dynamics_model = create_model(self.cfg, self.context_cfg, self.backbone_cfg)

        generator = torch.Generator(device=device)
        generator.manual_seed(0)

        def term_fn(action, state):
            done = torch.where(state[:, 0] < -2.4) \
                or torch.where(state[:, 0] > 2.4) \
                or torch.where(state[:, 2] < -(12 * 2 * math.pi / 360)) \
                or torch.where(state[:, 2] > (12 * 2 * math.pi / 360))
            done = bool(done)
            return done

        reward_fn = None

        # Create a gym-like environment to encapsulate the model
        self.model_env = ModelEnv(env, self.dynamics_model, term_fn, reward_fn, generator=generator)

        self.replay_buffer = common_util.create_replay_buffer(self.cfg, env.observation_space.shape, env.action_space.shape, rng=np.random.default_rng(seed=0))

        common_util.rollout_agent_trajectories(
            env,
            self.trial_length, # initial exploration steps
            planning.RandomAgent(env),
            {}, # keyword arguments to pass to agent.act()
            replay_buffer=self.replay_buffer,
            trial_length=self.trial_length
        )

        print("# samples stored", self.replay_buffer.num_stored)

        agent_cfg = omegaconf.OmegaConf.create({
            # this class evaluates many trajectories and picks the best one
            "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
            "planning_horizon": 15,
            "replan_freq": 1,
            "verbose": False,
            "action_lb": "???",
            "action_ub": "???",
            # this is the optimizer to generate and choose a trajectory
            "optimizer_cfg": {
                "_target_": "mbrl.planning.CEMOptimizer",
                "num_iterations": 5,
                "elite_ratio": 0.1,
                "population_size": 500,
                "alpha": 0.1,
                "device": device,
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": True,
            }
        })
        self.agent = planning.create_trajectory_optim_agent_for_model(
            self.model_env,
            agent_cfg,
            num_particles=20
        )

        # Create a trainer for the model
        self.model_trainer = models.ModelTrainer(self.dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

    def run(self, env):
        train_losses = []
        val_scores = []

        def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
            train_losses.append(tr_loss)
            val_scores.append(val_score.mean().item())   # this returns val score per ensemble model

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
                    self.dynamics_model.update_normalizer(self.replay_buffer.get_all())  # update normalizer stats

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
                next_obs, reward, done, info = env.step(action)
                self.replay_buffer.add(obs, action, next_obs, reward, done)

                obs = next_obs
                total_reward += reward
                steps_trial += 1

                if steps_trial == self.trial_length:
                    break

            all_rewards.append(total_reward)

        plt.figure(dpi=200)
        plt.xlabel("Trial")
        plt.ylabel("Trial reward")
        plt.plot(all_rewards, 'bs-')
        plt.savefig('rewards.png')

        return train_losses, val_scores


    def plot(self, train_losses, val_scores, path):
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        ax[0].plot(train_losses)
        ax[0].set_xlabel("Total training epochs")
        ax[0].set_ylabel("Training loss (avg. NLL)")
        ax[1].plot(val_scores)
        ax[1].set_xlabel("Total training epochs")
        ax[1].set_ylabel("Validation score (avg. MSE)")
        plt.savefig(path)
