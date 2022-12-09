import numpy as np
from .CartpoleBalance import CartPoleEnv_template
from .CartpoleSwingup import CartPoleSwingUpEnv_template

# context limits - [low, default, high] TODO: take from sunblaze_envs
CARTPOLE_BALANCE_LIMITS = {
    "masscart": [1.0],
    "masspole": [0.1],
    "polelength": [0.5],
    "gravity": [9.8],
    "force_mag": [10.0]
}

CARTPOLE_SWINGUP_LIMITS = {
    "masscart": [0.5],
    "masspole": [0.5],
    "polelength": [0.5],
    "gravity": [9.8],
    "force_mag": [20.0]
}


class ContextSampler:
    def __init__(self, params, rng: np.random.Generator) -> None:
        self.param_keys = list(params.keys())
        self.param_vals = list(params.values())
        self.selection = "choice" if len(self.param_vals[0]) > 2 else "random"
        self.rng = rng

    def sample(self):
        if self.selection == "choice":
            return {
                key: self.rng.choice(values) for key, values in zip(self.param_keys, self.param_vals) 
            }
        else:
            return {
                key: self.rng.uniform(values[0], values[1]) for key, values in zip(self.param_keys, self.param_vals) 
            }


class ContexualEnv():
    def __init__(self, config, rng) -> None:
        super().__init__()

        self.c_train = ContextSampler(config.train_params[0], rng)
        self.c_test = ContextSampler(config.test_range[0], rng)

        if config.env == 'CartpoleBalance':
            self.env = CartPoleEnv_template
        elif config.env == 'CartpoleSwingUp':
            self.env = CartPoleSwingUpEnv_template
        else:
            raise NotImplementedError

        context = self.c_train.sample()
        dummy_env = self.env(**context)
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

    def reset(self, train=True):
        if train:
            context = self.c_train.sample()
        else:
            context = self.c_test.sample()
        return self.env(**context), context


class DummyContextualEnv():
    """Wrapper over Gym.env to convert to Contextual Env.
    Returns orginal env at each sample"""
    def __init__(self, env, rng) -> None:
        super().__init__()
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, train=True):
        return self.env, None


if __name__=='__main__':
    import yaml
    from easydict import EasyDict

    seed = 0
    rng = np.random.default_rng(seed)

    with open('./configs/cartpole.yaml') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    env_fam = ContexualEnv(config, rng)

    print('#'*80)
    print('sampled training contexts')
    for _ in range(10):
        env, context = env_fam.reset(train=True)
        print(context)

    print('#'*80)
    print('sampled testing contexts')
    for _ in range(10):
        env, context = env_fam.reset(train=False)
        print(context)
        
    s = env.reset()
    done = False
    while not done:
        a = np.random.random()*2 - 1
        s, r, done, _ = env.step([a])
        print(s, r, done)