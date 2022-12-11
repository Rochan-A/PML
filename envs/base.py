import itertools

import numpy as np

from .CartpoleBalance import CartPoleEnv_template
from .CartpoleSwingup import CartPoleSwingUpEnv_template


class ContextSampler:
    """Class to assist with sampling contexts from env fam."""

    def __init__(
        self, params, deterministic=False, rng: np.random.Generator = None
    ) -> None:
        self.param_keys = list(params.keys())
        self.param_vals = list(params.values())
        self.deterministic = deterministic
        if not deterministic:
            assert rng is not None, "Missing rng param if you want random sampling!"
            self.selection = "choice" if len(self.param_vals[0]) > 2 else "random"
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

    def sample(self, idx=None):
        if not self.deterministic:
            return self._sample()
        else:
            assert (
                idx is not None
            ), "idx param needs to be passed for deterministic sampling!"
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


class ContexualEnv:
    def __init__(self, config, rng) -> None:
        super().__init__()

        self.c_train = ContextSampler(config.train_params[0], rng)
        self.c_test = ContextSampler(config.test_range[0], deterministic=True)

        if config.env == "CartpoleBalance":
            self.env = CartPoleEnv_template
        elif config.env == "CartpoleSwingUp":
            self.env = CartPoleSwingUpEnv_template
        else:
            raise NotImplementedError

        context = self.c_train.sample()
        dummy_env = self.env(**context)
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

    def reset(self, idx=None):
        if idx is None:
            context = self.c_train.sample()
        else:
            context = self.c_test.sample(idx)
        return self.env(**context), context

    def __len__(self):
        return len(self.c_test.permute_params)


class DummyContextualEnv:
    """Wrapper over Gym.env to convert to Contextual Env.
    Returns orginal env at each sample"""

    def __init__(self, env, rng) -> None:
        super().__init__()
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, idx=True):
        return self.env, None

    def __len__(self):
        return 10


if __name__ == "__main__":
    import yaml
    from easydict import EasyDict

    seed = 0
    rng = np.random.default_rng(seed)

    with open("./configs/cartpole.yaml") as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    env_fam = ContexualEnv(config, rng)

    print("#" * 80)
    print("sampled training contexts")
    for _ in range(10):
        env, context = env_fam.reset(idx=True)
        print(context)

    print("#" * 80)
    print("sampled testing contexts")
    for _ in range(10):
        env, context = env_fam.reset(idx=0)
        print(context)

    s = env.reset()
    done = False
    while not done:
        a = np.random.random() * 2 - 1
        s, r, done, _ = env.step([a])
        print(s, r, done)
