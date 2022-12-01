import gym

class ContexualEnv(gym.Wrapper):
    def __init__(self, env, config) -> None:
        super().__init__(env)