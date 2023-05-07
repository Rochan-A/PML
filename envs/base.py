import itertools

import numpy as np

from .CartpoleBalance import CartPoleEnv
from .CartpoleSwingup import CartPoleSwingUpEnv

ENVS = {"CartpoleBalance": CartPoleEnv, "CartpoleSwingup": CartPoleSwingUpEnv}


class ContextSampler:
    """Class to assist with generating env params for env fam.

    Args:
        params (dict): Dictionary of parameters to sample from.
        deterministic (bool, optional): If True, samples all possible combinations of params.
            Used for testing. Defaults to False.
        rng (np.random.Generator, optional): Random number generator. Defaults to None.
    """

    def __init__(
        self, params: dict, deterministic=False, rng: np.random.Generator = None
    ) -> None:
        # Get params names and their values
        self.param_keys = list(params.keys())
        self.param_vals = list(params.values())
        self.deterministic = deterministic

        if not deterministic:
            assert (
                rng is not None
            ), "Missing rng param for non-deterministic context sampling!"
            assert (
                len(self.param_vals[0]) >= 1
            ), "Need at least one value needed for each param!"
            # if params have 2 values, sample uniformly between [low, high] values
            # otherwise, sample from discrete set of values
            self.selection = "random" if len(self.param_vals[0]) == 2 else "choice"
            self.rng = rng
        else:
            self.permute_params = []
            for r in itertools.product(*self.param_vals):
                self.permute_params.append(
                    {
                        key: r[i]
                        for key, i in zip(self.param_keys, range(len(self.param_keys)))
                    }
                )

    def __call__(self, idx: int = None) -> dict:
        if not self.deterministic:
            return self._sample()
        else:
            assert (
                idx is not None
            ), "Missing idx param for deterministic context sampling!"
            return self.permute_params[idx]

    def _sample(self):
        if self.selection == "choice":
            return {
                key: self.rng.choice(values)
                for key, values in zip(self.param_keys, self.param_vals)
            }
        else:
            return {
                key: self.rng.uniform(values[0], values[1])
                for key, values in zip(self.param_keys, self.param_vals)
            }


class ContextEnv:
    """Class that wraps gym.Env to add context sampling.

    Args:
        config (EasyDict): Config object.
        rng (np.random.Generator): Random number generator.
        seed (int): Seed for rng.

    Methods:
        reset(idx=None): Resets the environment and returns the initial observation.
        len(): Returns the number of contexts for testing.
    """

    def __init__(self, *, config, rng, seed, **kwargs) -> None:
        super().__init__()

        assert config.env in ENVS.keys(), f"Env {config.env} not found!"
        self.env = ENVS[config.env]
        self.seed = seed

        self.c_train = ContextSampler(config.train_params[0], rng=rng)
        self.c_test = ContextSampler(config.test_range[0], deterministic=True)

        context = self.c_train()
        dummy_env = self.env(**context)
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

    def reset(self, idx=None):
        if idx is None:
            context = self.c_train()
        else:
            context = self.c_test(idx)
        env = self.env(**context)
        env.seed(self.seed)
        return env, context

    def __len__(self):
        return len(self.c_test.permute_params)


class DummyContextEnv:
    """Wrapper to convert standard gym.Env to ContextEnv. Does not change
    the underlying env, returns the same env at each sample.

    Args:
        env (gym.Env): Standard gym.Env to wrap.
        kwargs (dict): Additional kwargs to pass to ContextEnv.
            len (int): Number of contexts to sample.
    """

    def __init__(self, *, env, **kwargs) -> None:
        super().__init__()
        self.env = env
        self.len = kwargs.get("len", 10)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        return self.env, None

    def __len__(self):
        return self.len


if __name__ == "__main__":
    import yaml
    from easydict import EasyDict

    seed = 0
    rng = np.random.default_rng(seed)

    with open("../configs/cartpole.yaml") as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    env_fam = ContextEnv(config=config, rng=rng, seed=0)

    print("#" * 80)
    print("sampled training contexts")
    for idx in range(10):
        env, context = env_fam.reset(idx=idx)
        print(context)

    print("#" * 80)
    print("sampled testing contexts")
    for idx in range(10):
        env, context = env_fam.reset(idx=idx)
        print(context)

    s = env.reset()
    done = False
    while not done:
        a = np.random.random() * 2 - 1
        s, r, done, _ = env.step([a])
        print(s, r, done)
