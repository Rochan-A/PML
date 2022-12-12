import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf
from pprint import pprint
from os.path import join
from tqdm import tqdm, trange

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
        self.context_len = None if config.context.no_context else config.context.history_size

        if model is None:
            self.ensemble_size = config.transitionreward.ensemble_size

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
                        "num_layers": config.transitionreward.hidden_layers,
                        "ensemble_size": self.ensemble_size,
                        "hid_size": config.transitionreward.hidden_dim,
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

            # Contruct context, stateaction encoder configs
            self.context_cfg = config.context
            self.context_cfg["state_sz"] = env.observation_space.shape[0]
            self.context_cfg["action_sz"] = env.action_space.shape[0]
            self.stateaction_cfg = config.stateaction
            self.stateaction_cfg["state_sz"] = env.observation_space.shape[0]
            self.stateaction_cfg["action_sz"] = env.action_space.shape[0]

            # Create a dynamics model for this environment
            self.dynamics_model = create_model(
                cfg=self.cfg,
                context_cfg=self.context_cfg,
                stateaction_cfg=self.stateaction_cfg,
                eval_cfg=config.agent.eval,
                use_context=False if self.context_len is None else True
            )

            # load the model
            self.dynamics_model.load(args.load)

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


    def run(self, env_fam, env, num_trials, PATH):
        if self.context_len:
            rhc = RollingHistoryContext(
                K=self.context_len,
                state_sz=env.observation_space.shape[0],
                action_sz=env.action_space.shape[0]
            )

        all_rewards = []
        all_contexts = []

        # len(env_fam) returns the number of MDPs to test over
        # iterate over all possible MDP permutations
        for MDP in range(len(env_fam)):

            # Sample CMDP from distribution. If --mdp flag, then it returns the
            # same MDP.
            env, ctx_vals = env_fam.reset(idx=MDP)
            all_contexts.append(ctx_vals)
            print('trial: {}\t Context vector: {}'.format(MDP, ctx_vals if ctx_vals is not None else '<fixed>'))

            rewards = []
            for trial in trange(num_trials):

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
                rewards.append(total_reward)

            all_rewards.append(rewards)

        self.plot_single(
            np.mean(np.array(all_rewards), axis=-1),
            join(PATH,'test_rewards.png'),
            xlabel="Trial",
            ylabel="Reward"
        )
        return {
                'rewards': all_rewards,
                'all_contexts': all_contexts
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
