import torch, math
import numpy as np

def cartpole_upright_term(action, state):
    done = torch.where(state[:, 0] < -2.4) \
        or torch.where(state[:, 0] > 2.4) \
        or torch.where(state[:, 2] < -(12 * 2 * math.pi / 360)) \
        or torch.where(state[:, 2] > (12 * 2 * math.pi / 360))
    done = bool(done)
    return done

def cartpole_upright_reward(action, next_state):
    return 1 if not cartpole_upright_term(action, next_state) else 0

def cartpole_swingup_term(action, state):
    done = torch.where(state[:, 0] < -2.4) or torch.where(state[:, 0] > 2.4)
    done = bool(done)
    return done

class cartpole_swingup_rew:
    def __init__(self, l) -> None:
        self.l

    def __call__(self, action, next_state):
        x, theta, = next_state[:, 0], next_state[:, 2]
        length = self.l  # pole length
        x_tip_error = x - length * np.sin(theta)
        y_tip_error = length - length * np.cos(theta)
        reward = np.exp(-(x_tip_error ** 2 + y_tip_error ** 2) / length ** 2)
        return reward
