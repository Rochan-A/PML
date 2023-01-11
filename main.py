import yaml, math
from easydict import EasyDict
from os.path import join
import numpy as np
import argparse
from pprint import pprint

import bz2
import pickle
import _pickle as cPickle

import torch

from tensorboardX import SummaryWriter

import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns

from trainers import Trainer, Tester
from envs import ContexualEnv, DummyContextualEnv, configure_reward_fn, configure_term_fn
from utils.utils import without_keys, compress_pickle, decompress_pickle, make_dirs, gen_save_path

def train(args, config, PATH):
    """train

    Args
    ----
        args (argparse): cmd line args
        config (easydict): config read from file
        PATH (str): save path
    """

    rng = np.random.default_rng(args.seed)

    if args.mdp:
        # For testing implementation
        import gym
        env = cartpole_env.CartPoleEnv()
        env.seed(args.seed)
        env_fam = DummyContextualEnv(env, rng)

        # This functions allows the model to evaluate the true rewards given
        # an observation 
        reward_fn = reward_fns.cartpole
        # This function allows the model to know if an observation should make
        # the episode end
        term_fn = termination_fns.cartpole
    else:
        # Contextual env
        env_fam = ContexualEnv(config, rng)
        env, _ = env_fam.reset()

        term_fn = configure_term_fn(config)
        reward_fn = configure_reward_fn(config)

    writer = SummaryWriter(PATH)

    trainer_cfg = {
        'env': env,
        'env_fam': env_fam,
        'config': config,
        'reward_fn': reward_fn,
        'term_fn': term_fn,
        'args': args,
        'writer': writer,
        'no_test_flag': args.no_test_flag,
        'rng': rng,
        'save_path': PATH
    }
    print('Trainer_cfg')
    pprint(trainer_cfg)

    # Initialize Trainer
    algo = Trainer(**trainer_cfg)
    data = algo.run(env_fam, env, PATH)

    compress_pickle(join(PATH, 'train_data.pkl'), without_keys(data, 'model'))

    return data


def test(args, config, model=None):
    """test

    Args
    ----
        args (argparse): cmd line args
        config (easydict): config read from file
    """

    rng = np.random.default_rng(args.seed)

    if args.mdp:
        import gym
        env = cartpole_env.CartPoleEnv()
        env.seed(args.seed)
        env_fam = DummyContextualEnv(env)

        # This functions allows the model to evaluate the true rewards given
        # an observation 
        reward_fn = reward_fns.cartpole
        # This function allows the model to know if an observation should make
        # the episode end
        term_fn = termination_fns.cartpole
    else:
        # Contextual env
        env_fam = ContexualEnv(config, rng)
        env, _ = env_fam.reset()

        term_fn = configure_term_fn(config)
        reward_fn = configure_reward_fn(config)

    tester_cfg = {
        'env': env,
        'config': config,
        'reward_fn': reward_fn,
        'term_fn': term_fn,
        'model': model,
        'args': args
    }
    print('tester_cfg')
    pprint(tester_cfg)

    tester = Tester(**tester_cfg)
    data = tester.run(env_fam, env, 1, PATH)

    compress_pickle(join(PATH, 'test_data.pkl'), data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to test Generalization in Contextual MDPs for MBRL")
    parser.add_argument("--config", help="config file")
    parser.add_argument("--root", default="./saves/", help="root folder to save experiments")
    parser.add_argument("--exp-name", default=None, help="experiment name")
    parser.add_argument("--load", help="path to load models from")
    parser.add_argument("--logger", action="store_true", help="Print training log")
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

    PATH = gen_save_path(args, config)

    if args.no_test_flag:
        make_dirs(PATH)
        _ = train(args, config, PATH)
    elif args.only_test_flag:
        test(args, config)
    else:
        make_dirs(PATH)
        data = train(args, config, PATH)
        test(args, config, data['model'])