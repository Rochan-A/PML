import yaml, math
from easydict import EasyDict
from pathlib import Path
from os.path import join

import argparse, datetime

import torch

from tensorboardX import SummaryWriter

from trainers import Trainer
from envs import ContexualEnv, DummyContextualEnv

import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns

from envs.term_rew import *

def make_dirs(directory):
    """Make dir path if it does not exist
    
    Args
    ----
        path (str): path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def gen_save_path(args, config):
    """Generate save path

    Args
    ----
        args (argsparse): cmd line args
        config (easydict): config read from file

    Returns
    -------
        path (str)
    """

    PATH = join(args.root, "{}".format(config.env))

    if config.normalize_flag:
        PATH = join(PATH, "norm")
    else:
        PATH = join(PATH, "raw")


    # TODO: add params to path
    PATH = join(PATH, "seed_" + str(args.seed))
    DT = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    PATH = join(PATH, DT)

    print('#'*80)
    print("Saving to: {}".format(PATH))
    return PATH


def train(args, config, PATH):
    """train

    Args
    ----
        args (argparse): cmd line args
        config (easydict): config read from file
        PATH (str): save path
    """

    if args.mdp:
        import gym
        env = cartpole_env.CartPoleEnv()
        env.seed(args.seed)
        env_fam = DummyContextualEnv(env)

        # This functions allows the model to evaluate the true rewards given an observation 
        reward_fn = reward_fns.cartpole
        # This function allows the model to know if an observation should make the episode end
        term_fn = termination_fns.cartpole
    else:
        # Contextual env
        env_fam = ContexualEnv(config)
        env, _ = env_fam.reset()

        term_fn = cartpole_upright_term
        reward_fn = cartpole_upright_reward

    writer = SummaryWriter(PATH)

    # Initialize Trainer
    algo = Trainer(
        env=env,
        env_fam=env_fam,
        config=config,
        reward_fn=reward_fn,
        term_fn=term_fn,
        args=args,
        writer=writer,
        no_test_flag=args.no_test_flag,
        only_test_flag=args.only_test_flag
    )

    train_losses, val_scores, all_rewards = algo.run(env_fam, env, PATH)

    algo.plot(
        [train_losses, val_scores],
        join(args.root, 'plot.png'),
        ["Epoch", "Epoch"],
        ["Training loss (avg. NLL)", "Validation score (avg. MSE)"]
    )
    algo.plot_single(
        all_rewards,
        join(args.root,'rewards.png'),
        xlabel="Trial",
        ylabel="Reward"
    )


def test(args, config, PATH):
    """test

    Args
    ----
        args (argparse): cmd line args
        config (easydict): config read from file
        PATH (str): save path
    """

    if args.mdp:
        import gym
        env = cartpole_env.CartPoleEnv()
        env.seed(args.seed)
        env_fam = DummyContextualEnv(env)

        # This functions allows the model to evaluate the true rewards given an observation 
        reward_fn = reward_fns.cartpole
        # This function allows the model to know if an observation should make the episode end
        term_fn = termination_fns.cartpole
    else:
        # Contextual env
        env_fam = ContexualEnv(config)
        env, _ = env_fam.reset()

        term_fn = cartpole_upright_term
        reward_fn = cartpole_upright_reward

    writer = SummaryWriter(PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to test Generalization in Contextual MDPs for MBRL")
    parser.add_argument("--config", help="config file")
    parser.add_argument("--root", default="./saves/", help="experiments name")
    parser.add_argument("--seed", type=int, default=0, help="random_seed")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device idx")
    parser.add_argument(
        "--mdp", action="store_true", help="flag to enable only MDP training"
    )
    parser.add_argument(
        "--no_test_flag", action="store_true", help="flag to disable test"
    )
    parser.add_argument(
        "--only_test_flag", action="store_true", help="flag to enable only test"
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config['device'] = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
        config = EasyDict(config)
    else:
        raise ValueError(str(args.config))

    # generate save path
    PATH = gen_save_path(args, config)
    make_dirs(PATH)

    if args.no_test_flag:
        train(args, config, PATH)
    elif args.only_test_flag:
        raise NotImplementedError
        test(args, config, PATH)
    else:
        train(args, config, PATH)
        raise NotImplementedError
        test(args, config, PATH)