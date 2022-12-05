import torch, math
import numpy as np

def cartpole_upright_term(action, state):
    done =(state[:, 0] < -2.4) \
        * (state[:, 0] > 2.4) \
        * (state[:, 2] < -(12 * 2 * math.pi / 360)) \
        * (state[:, 2] > (12 * 2 * math.pi / 360))
    return done

def cartpole_upright_reward(action, next_state):
    ones = torch.ones((next_state.shape[0]), device=next_state.device)
    zeros = torch.zeros((next_state.shape[0]), device=next_state.device)
    done = (next_state[:, 0] < -2.4) \
        * (next_state[:, 0] > 2.4) \
        * (next_state[:, 2] < -(12 * 2 * math.pi / 360)) \
        * (next_state[:, 2] > (12 * 2 * math.pi / 360))
    k = torch.where(done, zeros, ones).reshape(-1, 1)
    return k

def cartpole_swingup_term(action, state):
    done = (state[:, 0] < -2.4) * (state[:, 0] > 2.4)
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
        return torch.tensor([reward], dtype=torch.float32, device=next_state.device)
