from os.path import join

import argparse
import yaml
from easydict import EasyDict

from utils.utils import decompress_pickle, make_plots


def load_experiments(paths):
    """load experiment data"""

    data = {'train': {}, 'test': {}}

    for path in paths:
        name = path[:-1].split('/')[-1]
        try:
            data['train'][name] = decompress_pickle(join(path, 'train_data.pkl'))
        except FileNotFoundError:
            print('Training data missing for {} ...'.format(path))

        try:
            data['test'][name] = decompress_pickle(join(path, 'test_data.pkl'))
        except FileNotFoundError:
            print('Testing data missing for {} ...'.format(path))

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Path to plotter config file", required=True)
    parser.add_argument("-p", "--paths", nargs='+', help="Path to experiment folders", type=str, required=True)
    parser.add_argument("-f", "--format", default="png", help="Format to save plots (pdf/png (default)).")
    parser.add_argument("-s", "--save", help="Path to save", required=True)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    data = load_experiments(args.paths)
    assert len(data['test']) > 0, "No data found, exiting..."

    make_plots(data, args.save, args.format)