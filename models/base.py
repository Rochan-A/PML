import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self, hidden_layer_size=256, state_size=128, out_dim=256):
        super(Backbone, self).__init__()

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(state_size + 1, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, out_dim)
        self.relu = nn.ReLU()

        self.linear_relu_stack = nn.Sequential(
            self.input_layer,
            self.relu,
            self.hidden_layer,
            self.relu,
            self.output_layer,
            self.relu,
        )

        self.fc_mu = nn.Linear(hidden_layer_size, out_dim)
        self.fc_var = nn.Linear(hidden_layer_size, out_dim)

    def forward(self, state_vector_t, action_t):
        x = torch.cat((state_vector_t, action_t), dim=-1)
        output = self.linear_relu_stack(x)

        mu = self.fc_mu(output)
        log_var = self.fc_var(output)
        return [mu, log_var, output]


class Head(nn.Module):
    def __init__(self, concat_vec_size, hidden_layer_size, state_size):
        super(Head, self).__init__()

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(concat_vec_size, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, state_size)

        self.linear_relu_stack = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            self.hidden_layer,
            nn.ReLU(),
            self.output_layer,
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(state_size, state_size)
        self.fc_var = nn.Linear(state_size, state_size)

    def forward(self, concat_vec):
        x = self.flatten(concat_vec)
        x = self.linear_relu_stack(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var, x]

class RewardModel(nn.Module):
    def __init__(self, state_size, hidden_layer_size, back_context_out):
        super(RewardModel, self).__init__()

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(state_size + back_context_out, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, 1)

        self.linear_relu_stack = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            self.hidden_layer,
            nn.ReLU(),
            self.output_layer,
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(1, 1)
        self.fc_var = nn.Linear(1, 1)

    def forward(self, next_states, back_context_out):
        x = torch.cat((next_states, back_context_out), dim=-1)
        x = self.linear_relu_stack(x)
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var, x]


class DNet(nn.Module):
    def __init__(self, env, config):
        super(DNet, self).__init__()

        state_sz = env.observation_space.shape[0]

        self.device = config.device

        self.context_enc = ContextEncoder(config.context.history_sz, state_sz, config.context.hidden_sizes, config.context.out_dim).to(device=self.device)
        self.backbone = Backbone(config.backbone.hidden_sizes, state_sz, config.backbone.out_dim).to(device=self.device)
        self.reward_enc = [RewardModel(state_sz, config.reward.model.hidden_sizes, config.backbone.out_dim+config.context.out_dim).to(device=self.device) for _ in range(config.head.ensemble_size)]
        self.heads = [Head(config.backbone.out_dim+config.context.out_dim, config.head.hidden_sizes, state_sz).to(device=self.device) for _ in range(config.head.ensemble_size)]

    def forward(self, state, action, history):
        b_embb = self.backbone.forward(state, action)
        c_embb = self.context(history)
        embb = torch.cat([b_embb, c_embb], dim=-1)
        m_head_out = []
        for head in self.heads:
            m_head_out.append(head.forward(embb))
        return b_embb, c_embb, m_head_out

    def reward(self, state, action, history=None):
        b_embb, c_embb, pred_next_state = self.forward(state, action, history)
        bc_embb = torch.cat([b_embb, c_embb], dim=-1)
        pred_rews = []
        for idx in range(len(self.heads)):
            pred_rews.append(self.reward_enc[idx].forward(pred_next_state[idx], bc_embb))
        return pred_rews

    def context(self, history):
        return self.context_enc.forward(history)



if __name__ == "__main__":
    net = Backbone()
    print(net.forward(torch.zeros((1, 128)), torch.zeros((1, 1))).shape)

    net = ContextEncoder()
    print(net.forward(torch.zeros((1, 129 * 16))).shape)

    net = Head(128, 128, 128)
    print(net.forward(torch.zeros((1, 128))).shape)

    net = RewardModel(128, 128, 128)
    print(net.forward(torch.zeros((1, 128)), torch.zeros((1, 128))).shape)

    import gym
    import sunblaze_envs

    import yaml
    from easydict import EasyDict

    with open('./configs/cartpole.yaml') as f:
        config = yaml.safe_load(f)
        config['device'] = torch.device('cpu')
    config = EasyDict(config)

    env = sunblaze_envs.make('SunblazeCartPole-v0')
    state_sz = env.observation_space.shape[0]

    net = DNet(env, config)

    with torch.no_grad():
        b_embb, c_embb, m_head_out = net.forward(torch.zeros((1, state_sz)),
                            torch.zeros((1, 1)),
                            torch.zeros((1, config.context.history_sz*(state_sz + 1)))
                            )
        print("backbone_embb: {}\nhead_out: {}\ncontext_embb: {}\nreward: {}".format(
                b_embb.shape,
                m_head_out[0].shape,
                net.context(torch.zeros((1, config.context.history_sz*(state_sz + 1)))
                            ).shape,
                net.reward(torch.zeros((1, state_sz)),
                            torch.zeros((1, 1)),
                            torch.zeros((1, config.context.history_sz*(state_sz + 1)))
                            )[0].shape
                ))