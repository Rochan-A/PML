import torch, math

def cartpole_upright_term(action, state):
    ones = torch.ones((state.shape[0]), device=state.device)
    zeros = torch.zeros((state.shape[0]), device=state.device)
    done = torch.where(state[:, 0] < -2.4, ones, zeros) + torch.where(state[:, 0] > 2.4, ones, zeros) \
        + torch.where(state[:, 2] < -(12 * 2 * math.pi / 360), ones, zeros) + torch.where(state[:, 0] > (12 * 2 * math.pi / 360), ones, zeros)
    done = torch.where(done >= 1, ones, zeros).reshape(-1, 1) > 0
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

def cartpole_swingup_reward(action, next_state):
    x, cos_theta, sin_theta = next_state[:, 0], next_state[:, 2], next_state[:, 3]
    theta = 2*torch.arctan(sin_theta/(1 + cos_theta))
    length = 0.6
    x_tip_error = x - length * torch.sin(theta)
    y_tip_error = length - length * torch.cos(theta)
    reward = torch.exp(-(x_tip_error ** 2 + y_tip_error ** 2) / length ** 2)
    return reward.reshape(-1, 1)


def configure_reward_fn(config):
    if not config.reward_fn:
        return None
    if config.env == 'CartpoleSwingUp':
        return cartpole_swingup_reward
    elif config.env == 'CartpoleBalance':
        return cartpole_upright_reward
    else:
        raise NotImplementedError

def configure_term_fn(config):
    if config.env == 'CartpoleSwingUp':
        return cartpole_swingup_term
    elif config.env == 'CartpoleBalance':
        return cartpole_upright_term
    else:
        raise NotImplementedError