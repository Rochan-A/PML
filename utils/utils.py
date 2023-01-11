import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import datetime

import bz2
import pickle
import _pickle as cPickle

from pathlib import Path

import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

sns.set_style('darkgrid')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=MEDIUM_SIZE)


def without_keys(d, *keys):
     return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))


def compress_pickle(fname, data):
    with bz2.BZ2File(fname, 'wb') as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


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

    PATH = join(PATH, "seed_" + str(args.seed))
    DT = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") if args.exp_name is None else args.exp_name
    PATH = join(PATH, DT)

    print('#'*80)
    print("Saving to: {}".format(PATH))
    return PATH


def make_test_reward_plots(data, path, fmt):
    # names of different runs
    labels = list(data.keys())

    # get number of parameters
    n = len(data[labels[0]]['all_contexts'])

    fig, axs = plt.subplots(nrows=1, ncols=n, dpi=200, figsize=(8, 2))

    lgnd = []
    clrs = sns.color_palette('husl', n_colors=n)
    for idx in range(n):
        bp = axs[idx].boxplot(
            [data[label]['rewards'][idx] for label in labels],
            showfliers=False,
            patch_artist=True
        )
        for patch, color in zip(bp['boxes'], clrs):
            patch.set_facecolor(color)
        title = ""
        for key in data[labels[0]]['all_contexts'][idx].keys():
            title += "{}={}\n".format(
                key,
                data[labels[0]]['all_contexts'][idx][key]
            )
        axs[idx].set_title(title[:-1])
        axs[idx].set_xticklabels([])
        axs[idx].yaxis.set_major_locator(MaxNLocator(integer=True))

    for idx, label in enumerate(labels):
        lgnd.append(
            mpatches.Patch(color=clrs[idx], label=label)
        )
    axs[0].set_ylabel('Reward')
    plt.tight_layout()
    plt.legend(handles=lgnd, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.savefig(join(path, 'rewards.{}'.format(fmt)), format=fmt, bbox_inches='tight')


def make_plots(data, path, fmt='png'):
    make_dirs(path)
    make_test_reward_plots(data['test'], path, fmt=fmt)