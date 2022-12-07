import pickle as pkl
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

import time
from typing import Callable, List, Optional, Sequence, cast

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions

import mbrl.models
import mbrl.types
import mbrl.util.math

from mbrl.planning.core import Agent
from mbrl.planning.trajectory_opt import Optimizer, TrajectoryOptimizer
from .replay_buffer import get_basic_buffer_iterators, create_replay_buffer

class RollingHistoryContext:
    def __init__(self, K, state_sz, action_sz) -> None:
        self.K = K
        
        self.state_sz = state_sz
        self.action_sz = action_sz
        self.default_sz = (state_sz+action_sz)*K
        self.store = None
        self.prev_st = None

    def append(self, state, action=None):
        if self.prev_st is None:
            self.prev_st = state
        else:
            state = state - self.prev_st

        if action is None:
            action = np.random.rand(self.action_sz)

        k = np.concatenate([state, action], axis=0)
        if self.store is None:
            self.store = np.repeat(k, self.K)
        else:
            self.store = np.roll(self.store, -k.shape[0])
            self.store[-k.shape[0]:] = k

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
        self, obs: np.ndarray, context=None, optimizer_callback: Optional[Callable] = None, **_kwargs
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
        env_fam,
        reward_fn,
        term_fn,
        config,
        args,
    ):

        self.trial_length = config.trail_length
        self.num_trials = config.num_trials
        self.ensemble_size = config.head.ensemble_size
        self.context_len = None if config.context.no_context else config.context.history_size

        generator = torch.Generator(device=config["device"])
        generator.manual_seed(args.seed)

        # Everything with "???" indicates an option with a missing value.
        # Our utility functions will fill in these details using the
        # environment information
        in_sz = config.context.out_dim + config.backbone.out_dim if self.context_len is not None else env.observation_space.shape[0]+env.action_space.shape[0]
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

        agent_cfg = omegaconf.OmegaConf.create(
            {
                # this class evaluates many trajectories and picks the best one
                "_target_": "trainers.TrajectoryOptimizerAgent",
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


    def run(self, env_fam, env):

        if self.context_len:
            rhc = RollingHistoryContext(self.context_len, env.observation_space.shape[0], env.action_space.shape[0])

        all_rewards = [0]
        for trial in range(self.num_trials):

            # Sample CMDP from distribution. If --mdp flag, then it returns the
            # same MDP.
            env, ctx_vals = env_fam.reset(train=False)
            print('Context vector: {}'.format(ctx_vals if ctx_vals is not None else '<fixed>'))

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

            #NEEDS EDITS
            #data_file = ""
            #save_to_file(self, total_reward, data_file)

        return all_rewards

    def save_to_file(self, total_reward, data_file):

        data_dict = {}
        data_dict['total_reward'] = total_reward

        with open(data_file, 'wb') as f:
            pkl.dump(data_dict, f)






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
