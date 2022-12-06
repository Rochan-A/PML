import torch, math
import numpy as np

def cartpole_upright_term(action, state):
    done = (state[:, 0] < -2.4) \
        or (state[:, 0] > 2.4) \
        or (state[:, 2] < -(12 * 2 * math.pi / 360)) \
        or (state[:, 2] > (12 * 2 * math.pi / 360))
    return done

def cartpole_upright_reward(action, next_state):
    ones = torch.ones((next_state.shape[0]), device=next_state.device)
    zeros = torch.zeros((next_state.shape[0]), device=next_state.device)
    done = (next_state[:, 0] < -2.4) \
        or (next_state[:, 0] > 2.4) \
        or (next_state[:, 2] < -(12 * 2 * math.pi / 360)) \
        or (next_state[:, 2] > (12 * 2 * math.pi / 360))
    k = torch.where(done, zeros, ones).reshape(-1, 1)
    return k

def cartpole_swingup_term(action, state):
    ones = torch.ones((state.shape[0]), device=state.device)
    zeros = torch.zeros((state.shape[0]), device=state.device)
    done = torch.where(state[:, 0] < -2.4, ones, zeros) + torch.where(state[:, 0] > 2.4, ones, zeros)
    done = torch.where(done >= 1, ones, zeros).reshape(-1, 1) > 0
    return done

class cartpole_swingup_rew:
    def __init__(self, l) -> None:
        self.l = l

    def __call__(self, action, next_state):
        x, cos_theta, sin_theta = next_state[:, 0], next_state[:, 2], next_state[:, 3]
        theta = 2*torch.arctan(sin_theta/(1 + cos_theta))
        length = self.l  # pole length
        x_tip_error = x - length * torch.sin(theta)
        y_tip_error = length - length * torch.cos(theta)
        reward = torch.exp(-(x_tip_error ** 2 + y_tip_error ** 2) / length ** 2)
        return reward.reshape(-1, 1)
