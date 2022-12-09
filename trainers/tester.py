import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf
from pprint import pprint
from os.path import join

from models.transitionreward import create_model
from models.dynamics_model import ModelEnv

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions

import mbrl.models
import mbrl.types
import mbrl.util.math

from .trainer import RollingHistoryContext


def create_agent(agent_cfg, model_env, config):
    mbrl.planning.complete_agent_cfg(model_env, agent_cfg)
    agent = hydra.utils.instantiate(agent_cfg)

    def trajectory_eval_fn(initial_state, action_sequences, initial_context=None):
        return model_env.evaluate_action_sequences(
            action_sequences,
            initial_state=initial_state,
            num_particles=config.agent.max_particles,
            initial_context=initial_context
        )

    agent.set_trajectory_eval_fn(trajectory_eval_fn)
    return agent


class Tester(object):
    def __init__(
        self,
        env,
        config,
        reward_fn,
        term_fn,
        args,
        model
    ):

        self.trial_length = config.trail_length
        self.num_trials = config.num_trials
        generator = torch.Generator(device=config["device"])
        generator.manual_seed(args.seed)

        if model is None:
            self.ensemble_size = config.head.ensemble_size
            self.context_len = None if config.context.no_context else config.context.history_size


            # Everything with "???" indicates an option with a missing value.
            # Our utility functions will fill in these details using the
            # environment information
            in_sz = config.context.out_dim+config.stateaction.out_dim \
                if self.context_len is not None else \
                    env.observation_space.shape[0]+env.action_space.shape[0]
            cfg_dict = {
                # dynamics model configuration
                "dynamics_model": {
                    "model": {
                        "_target_": "mbrl.models.GaussianMLP",
                        "device": str(config["device"]),
                        "num_layers": config.head.hidden_layers,
                        "ensemble_size": self.ensemble_size,
                        "hid_size": config.head.hidden_dim,
                        "in_size": in_sz,
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
                self.cfg, self.context_cfg, self.backbone_cfg, False if self.context_len is None else True
            )

            # load the model
            self.dynamics_model.load(args.load)

            # Create custom gym-like environment to encapsulate the model TODO:
            self.model_env = ModelEnv(
                env, self.dynamics_model, term_fn, reward_fn, generator=generator
            )
        else:
            self.model_env = model

        agent_cfg = {
                # this class evaluates many trajectories and picks the best one
                "_target_": "trainers.TrajectoryOptimizerAgent",
                "planning_horizon": config.agent.horizon,
                "replan_freq": config.agent.replan_freq,
                "verbose": False,
                "action_lb": [-1.],
                "action_ub": [1.],
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
        print('Agent Optimizer cfg')
        pprint(agent_cfg)
        agent_cfg = omegaconf.OmegaConf.create(agent_cfg)

        # create agent
        self.agent = create_agent(agent_cfg, self.model_env, config)


    def run(self, env_fam, env, PATH):
        if self.context_len:
            rhc = RollingHistoryContext(
                K=self.context_len,
                state_sz=env.observation_space.shape[0],
                action_sz=env.action_space.shape[0]
            )

        all_rewards = [0]
        for trial in range(self.num_trials):

            # Sample CMDP from distribution. If --mdp flag, then it returns the
            # same MDP.
            env, ctx_vals = env_fam.reset(train=False)
            print('trial: {}\t Context vector: {}'.format(trial, ctx_vals if ctx_vals is not None else '<fixed>'))

            obs = env.reset()
            if self.context_len:
                rhc.reset()
                rhc.append(obs, None)

            self.agent.reset()

            done = False
            total_reward = 0.0
            steps_trial = 0

            while not done:

                if self.context_len:
                    action = self.agent.act(obs, rhc.store, **{})
                    rhc.append(obs, action)
                else:
                    action = self.agent.act(obs, **{})
                next_obs, reward, done, _ = env.step(action)

                obs = next_obs
                total_reward += reward
                steps_trial += 1

                if steps_trial == self.trial_length:
                    break
            all_rewards.append(total_reward)

        self.plot_single(
            all_rewards,
            join(PATH,'test_rewards.png'),
            xlabel="Trial",
            ylabel="Reward"
        )
        return {
                'rewards': all_rewards
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
        plt.ylabel(ylabel)
        plt.plot(data, "bs-")
        plt.savefig(path)
