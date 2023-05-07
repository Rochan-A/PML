import argparse
import sys
from os.path import join
from pprint import pprint

import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns

from torch.utils.tensorboard import SummaryWriter

from pets import PETSTrainer, PETSTester

from utils import (
    Logger,
    compress_pickle,
    filter_keys,
    gen_save_path,
    load_config,
    set_seeds,
)
from envs import ContextEnv, DummyContextEnv, configure_reward_fn, configure_term_fn

TRAINERS = {
    "pets": PETSTrainer,
}

TESTER = {
    "pets": PETSTester,
}


def train(args: argparse.Namespace, config: dict, PATH: str) -> dict:
    """Train model

    Args:
        args (argparse.Namespace): cmd line args
        config (dict): config read from file
        PATH (str): save path

    Returns:
        dict: data with keys: train_losses, val_scores, rewards, model, all_contexts
    """
    rng = set_seeds(args.seed)

    if args.dry_run:
        # For testing implementation
        dummy_env = cartpole_env.CartPoleEnv()
        dummy_env.seed(args.seed)
        env_fam = DummyContextEnv(env=dummy_env, rng=rng)

        # This functions allows the model to evaluate the true rewards given
        # an observation
        reward_fn = reward_fns.cartpole
        # This function allows the model to know if an observation should make
        # the episode end
        term_fn = termination_fns.cartpole
    else:
        # Contextual env
        env_fam = ContextEnv(config=config, rng=rng, seed=args.seed)
        dummy_env, _ = env_fam.reset()

        term_fn = configure_term_fn(config)
        reward_fn = configure_reward_fn(config)

    writer = SummaryWriter(log_dir=PATH, flush_secs=10)

    print("> Initializing Trainer...")
    algo = TRAINERS[args.method](
        dummy_env=dummy_env,
        env_fam=env_fam,
        reward_fn=reward_fn,
        term_fn=term_fn,
        config=config,
        rng=rng,
        save_path=PATH,
    )

    print("> Running Trainer...")
    data = algo.run(writer=writer)

    print("> Saving trainer data...")
    # Remove model from data and save
    compress_pickle(join(PATH, "train_data.pkl"), filter_keys(data, "model"))

    return data


def test(args: argparse.Namespace, config: dict, model=None) -> None:
    """Test model

    Args
    ----
        args (argparse.Namespace): cmd line args
        config (dict): config read from file
        model (nn.Module): model to test. If None, load from save_path.
    """
    rng = set_seeds(args.seed)

    if args.dry_run:
        # For testing implementation
        dummy_env = cartpole_env.CartPoleEnv()
        dummy_env.seed(args.seed)
        env_fam = DummyContextEnv(env=dummy_env, rng=rng)

        reward_fn = reward_fns.cartpole
        term_fn = termination_fns.cartpole
    else:
        # Contextual env
        env_fam = ContextEnv(config=config, rng=rng, seed=args.seed)
        dummy_env, _ = env_fam.reset()

        term_fn = configure_term_fn(config)
        reward_fn = configure_reward_fn(config)

    print("> Initializing Tester...")
    tester = TESTER[args.method](
        dummy_env=dummy_env,
        env_fam=env_fam,
        reward_fn=reward_fn,
        term_fn=term_fn,
        config=config,
        rng=rng,
        save_path=PATH,
        load_path=args.load,
        model=model,
    )

    print("> Running Tester...")
    data = tester.run(num_trials=config.get("num_test_trials", 10))

    print("> Saving tester data...")
    compress_pickle(join(PATH, "test_data.pkl"), data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to test Generalization of Model-Based RL in Contextual MDPs."
    )
    parser.add_argument(
        "-c", "--config", required=True, help="path to config config file"
    )
    parser.add_argument(
        "-m",
        "--method",
        required=True,
        help="Method to use.",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        default="./saves/",
        help="Path to save models and results. Default: ./saves/",
    )
    parser.add_argument("-e", "--exp-name", default=None, help="Experiment name")
    parser.add_argument("-l", "--load", help="Path to load models from.")
    parser.add_argument("--seed", type=int, default=0, help="random_seed")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device idx")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Flag to enable training for a single context. For testing implementation.",
    )
    parser.add_argument(
        "--no_test_flag", action="store_true", help="Flag to disable test"
    )
    parser.add_argument(
        "--only_test_flag", action="store_true", help="Flag to enable only test"
    )
    args = parser.parse_args()

    config = load_config(args)
    PATH = gen_save_path(args, config)
    sys.stdout = Logger(join(PATH, "log.txt"))

    pprint(args)
    pprint(config)

    if args.no_test_flag:
        print("> Running only train loop...")
        _ = train(args, config, PATH)
    elif args.only_test_flag:
        print("> Running only test loop...")
        test(args, config)
    else:
        print("> Running train & test loop...")
        data = train(args, config, PATH)
        test(args, config, data["model"])
