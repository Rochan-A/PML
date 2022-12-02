import yaml
from easydict import EasyDict
from pathlib import Path
from os.path import join

import argparse, datetime

import torch

from tensorboardX import SummaryWriter

from models import Dynamics
from controllers.mpc_controller import MPC
from trainers import Trainer
from envs import ContexualEnv


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

    PATH = join(args.root, "{}".format(args.env))

    if config.normalize_flag:
        PATH = join(PATH, "norm")
    else:
        PATH = join(PATH, "raw")

    if config.MPC.optimizer in ["random", "CEM"]:
        PATH = join(PATH, "{}".format(config.MPC.optimizer))
        # TODO: add optimizer params to path
        # PATH = join(PATH, "hor_{}".format(config.MPC.))
    else:
        raise ValueError(args.policy_type)

    # TODO: add params to path
    PATH = join(PATH, "seed_" + str(args.seed))
    DT = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    PATH = join(PATH, DT)

    print('#'*80)
    print("Saving to: {}".format(PATH))
    return PATH


def main(args, config, PATH):
    """Main function
    
    Args
    ----
        args (argparse): cmd line args
        config (easydict): config read from file
        PATH (str): save path
    """

    # Contextual env
    env = ContexualEnv(config)

    # Initialize Dynamics
    dynamics_model = Dynamics(
        env=env,
        config=config
    )

    # Initialize Policy
    policy = MPC(
        env=env,
        dynamics_model=dynamics_model,
        config=config
    )

    writer = SummaryWriter(PATH)

    # Initialize Trainer
    algo = Trainer(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        config=config,
        writer=writer,
        no_test_flag=args.no_test_flag,
        only_test_flag=args.only_test_flag
    )

    algo.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", help="config file")
    parser.add_argument("--root", default="./saves/", help="experiments name")
    parser.add_argument("--seed", type=int, default=0, help="random_seed")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device idx")
    parser.add_argument("--env", default="SunblazeCartPole-v0", help="environment name")
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
    make_dirs(PATH)

    main(args, config, PATH)
