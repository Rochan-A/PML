import bz2
import datetime
import pathlib
import pickle
import yaml
from easydict import EasyDict
import sys
import time
from os.path import join
from pathlib import Path
from typing import Any, Tuple

import _pickle as cPickle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

import torch
from torch.nn import Module

EPS = 1e-8

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

sns.set_style("darkgrid")
plt.style.use("seaborn-v0_8-darkgrid")

plt.rcParams["legend.frameon"] = True
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def set_seeds(seed: int) -> np.random.Generator:
    """Set seeds for reproducibility

    Args:
        seed (int): seed

    Returns:
        np.random.Generator: numpy random generator
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # False for reproducibility
    return np.random.default_rng(seed)


def load_config(args) -> dict:
    """Load config from path

    Args:
        args (argparse): cmd line args

    Returns:
        dict: config
    """
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["device"] = (
        torch.device("cuda:{}".format(args.cuda))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    config = EasyDict(config)
    return config


def filter_keys(d: dict, *keys: list[str]) -> dict:
    """Filter out keys from dictionary

    Args:
        d (dict): dictionary
        keys (list[str]): keys to filter out

    Returns:
        dict: filtered dictionary
    """
    return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))


def get_parameter_count(net: Module) -> int:
    """Get number of parameters in model

    Args:
        net (Module): torch.nn.Module

    Returns:
        int: number of parameters
    """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def make_dirs(path: str):
    """Make path if it does not exist, otherwise do nothing

    Args:
        path (str): path to generate
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def compress_pickle(fname: str, data: Any):
    """Compress data as pickle file and store at path

    Args:
        fname (str): Path to file with name
        data (Any)
    """
    with bz2.BZ2File(fname, "wb") as f:
        cPickle.dump(data, f)


def decompress_pickle(fname: str):
    """Decompress pickle file at path and return data

    Args:
        fname (str): path to file with name

    Returns:
        data
    """
    data = bz2.BZ2File(fname, "rb")
    data = cPickle.load(data)
    return data


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

    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DT = dt if args.exp_name is None else f"{args.exp_name}-{dt}"

    PATH = join(
        args.save_path,
        str(config.env),
        "norm" if config.normalize_flag else "raw",
        "seed_" + str(args.seed),
        DT,
    )
    print("> Saving to: {}".format(PATH))
    make_dirs(PATH)
    return PATH


def multi_image_plot(
    imgs: list[np.ndarray],
    titles: list[str],
    path: str,
    figsize: Tuple[int, int] = (15, 6),
    dpi: int = 200,
) -> None:
    """Plot multiple images in a single row.

    Args:
        imgs (list[np.ndarray]): List of images to plot.
        titles (list[str]): Titles for each image.
        path (str): Path to save the plot.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (15, 6).
        dpi (int, optional): Figure dpi. Defaults to 200.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    for i in range(len(imgs)):
        ax = plt.subplot(1, len(imgs), i + 1)
        ax.imshow(imgs[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i])

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show()


class Logger(object):
    """Logging class to log to both stdout and file
    source: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

    Usage: Simply call `sys.stdout = Logger("path/to/file")`
    Then, any `print()` statements will be logged to both stdout and file
    """

    def __init__(self, filename: str, mode: str = "a"):
        """Initialize logger

        Args:
            filename (str): path to file
            mode (str, optional): mode to open file. Defaults to "a".
        """
        self.terminal = sys.stdout
        try:
            self.log = open(filename, mode)
        except FileNotFoundError:
            self.log = open(filename, "x")

    def write(self, message):
        """Write to both stdout and file"""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    """Time a block of code
    Usage:
        `with Timer("name"):`

        `    # code to time`
    """

    def __init__(self, name: str):
        """Initialize timer with name

        Args:
            name (str): name of the timer
        """
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print(
            "====> [{}] Time: {:7.3f}s or {}".format(
                self.name, self.elapsed, time.strftime("%H:%M:%S", self.elapsedH)
            )
        )


def make_box_plots(
    data: dict[str, np.ndarray],
    path: str,
    fmt: str = "png",
    figsize: Tuple[int, int] = (15, 6),
    dpi: int = 200,
):
    """Make box plots for data

    Args:
        data (dict[str, np.ndarray]): Data to plot
        path (str): Path to save the plot
        fmt (str, optional): Format to save plot figure. Defaults to "png".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (15, 6).
        dpi (int, optional): Defaults to 200.
    """
    # names of different runs
    labels = list(data.keys())

    # get number of parameters
    n = len(data[labels[0]]["all_contexts"])

    _, axs = plt.subplots(nrows=1, ncols=n, dpi=dpi, figsize=figsize)

    lgnd = []
    clrs = sns.color_palette("husl", n_colors=n)
    for idx in range(n):
        bp = axs[idx].boxplot(
            [data[label]["rewards"][idx] for label in labels],
            showfliers=False,
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], clrs):
            patch.set_facecolor(color)
        title = ""
        for key in data[labels[0]]["all_contexts"][idx].keys():
            title += "{}={}\n".format(key, data[labels[0]]["all_contexts"][idx][key])
        axs[idx].set_title(title[:-1])
        axs[idx].set_xticklabels([])
        axs[idx].yaxis.set_major_locator(MaxNLocator(integer=True))

    for idx, label in enumerate(labels):
        lgnd.append(mpatches.Patch(color=clrs[idx], label=label))
    axs[0].set_ylabel("Reward")
    plt.tight_layout()
    plt.legend(
        handles=lgnd, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0
    )
    plt.savefig(join(path, "rewards.{}".format(fmt)), format=fmt, bbox_inches="tight")
