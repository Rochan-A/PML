# Generalization via Online Continual Adaptation in Model-based RL

## Setup

First, [install mbrl](https://github.com/facebookresearch/mbrl-lib) and all other [requirements](https://github.com/Rochan-A/PML/blob/master/requirements.txt).

## Usage

To run a set of tasks, use the below template with the path to the config file you want to use.

```bash
$ python main.py -c <path/to/config/file>
```

Additional command line options are
```bash
$ python main.py -h
usage: main.py [-h] [--config CONFIG] [--root ROOT] [--load LOAD] [--logger] [--seed SEED] [--cuda CUDA] [--mdp] [--no_test_flag] [--only_test_flag]

Code to test Generalization in Contextual MDPs for MBRL

optional arguments:
  -h, --help        show this help message and exit
  --config CONFIG   config file
  --root ROOT       experiments name
  --load LOAD       path to load models from
  --logger          Print training log
  --seed SEED       random_seed
  --cuda CUDA       CUDA device idx
  --mdp             flag to enable only MDP training
  --no_test_flag    flag to disable test
  --only_test_flag  flag to enable only test
```

## Config Files

We use YAML config files to keep track of experimental paramters. Their structure is as following.
```yaml
# Environment configuration
# env name
env: CartpoleSwingUp

# contexts for training
train_params:
- polelength: [0.4, 0.6, 0.8, 1., 1.2]
  masspole: [0.75, 1.0, 1.25]
  # masscart: [0.25, 0.5, 1.5, 2.5]
  # gravity: [8.8, 12.0]
  # force_mag: [18.0, 22.0]

# model contexts for testing
test_range:
- polelength: [0.2, 0.5, 0.9, 1.4]
  masspole: [0.7, 0.9, 1.4]
  # masscart: [1.5, 2.5]
  # gravity: [9.8, 12.8]
  # force_mag: [10.0, 15.0]
# normalize input (keep False, gives better generalization performance)
normalize_flag: False

# max len of episode
trail_length: 200
# Number of initial trials to fill replay buffer (wo/ agent actions)
init_trials: 5
# Number of episodes w/ agent actions
num_trials: 5
replay_buffer_sz: 10000

# Use reward function or not (keep false)
reward_fn: False

# Agent configuration
agent:
  # how long of the horizon to predict
  horizon: 50
  # how many random samples for mpc
  popsize: 200
  alpha: 0.1
  # freq of planning
  replan_freq: 1
  # iters per planning step
  num_iters: 5
  # fraction of pop to keep for next step
  elite_ratio: 0.1
  max_particles: 20
  # Properties of functions used to evaluate action sequence
  eval:
    # MC updates to take when updating the model
    mc_update: 2
    # Method to use 'default', 'greedy', 'kl', 'combine'
    method: kl
    # Samples from context to consider
    mc_samples: 50
    # Weight for pred_reward
    pred_weight: 1.
    # Weight for kl_distance
    kl_weight: 1.

# Model arch parameters
context:
  history_size: 50
  hidden_dim: 200
  hidden_layers: 1
  out_dim: 16
  hidden_actv: relu
  output_actv: identity
  no_context: False

stateaction:
  hidden_dim: 200
  hidden_layers: 1
  out_dim: 256
  hidden_actv: relu
  state_actv: identity
  action_actv: tanh

transitionreward:
  hidden_dim: 200
  hidden_layers: 3
  ensemble_size: 5
  prop_method: fixed_model
  actv: torch.nn.LeakyReLU

# Dynamics model training config
dynamics:
  learning_rate: 0.001
  batch_size: 256
  # ratio of validation set
  validation_ratio: 0.05
  epochs_per_step: 50
  # number of epochs to early terminate if loss doesn't decrease by much
  # (for potentially faster training)
  patience: 20
```
