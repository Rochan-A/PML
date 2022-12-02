import random
from .CartpoleBalance import CartPoleEnv_template

# context limits - [low, default, high] TODO: take from sunblaze_envs
CARTPOLE_BALANCE_LIMITS = {
    "masscart": [0.5, 1.0, 2.0],
    "masspole": [0.05, 0.1, 0.2],
    "polelength": [0.3, 0.5, 0.7],
    "gravity": [6.8, 9.8, 12.8],
    "force_mag": [5.0, 10.0, 15.0]
}


class ContextSampler:
    def __init__(self, params) -> None:
        self.param_keys = list(params.keys())
        self.param_vals = list(params.values())
        self.selection = "choice" if len(self.param_vals[0]) > 2 else "random"

    def sample(self):
        if self.selection == "choice":
            return {
                key: random.choice(values) for key, values in zip(self.param_keys, self.param_vals) 
            }
        else:
            return {
                key: random.uniform(values[0], values[1]) for key, values in zip(self.param_keys, self.param_vals) 
            }


class ContexualEnv():
    def __init__(self, config) -> None:
        super().__init__()

        self.c_train = ContextSampler(config.train_params[0])
        self.c_test = ContextSampler(config.test_range[0])

        if config.env == 'CarpoleBalance':
            self.env = CartPoleEnv_template
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


if __name__=='__main__':
    import yaml
    from easydict import EasyDict

    with open('./configs/cartpole.yaml') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    env_fam = ContexualEnv(config)

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
        a = random.random()*2 - 1
        s, r, done, _ = env.step([a])
        print(s, r, done)