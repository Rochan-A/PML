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

    def forward(self, state_vector_t, action_t):
        x = torch.cat((state_vector_t, action_t), dim=-1)
        output = self.linear_relu_stack(x)
        return output


class ContextEncoder(nn.Module):
    def __init__(
        self, history_size=16, state_size=128, hidden_layer_size=1024, context_len=128
    ):
        super(ContextEncoder, self).__init__()

        self.input_layer = nn.Linear((state_size + 1) * history_size, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, context_len)

        self.relu = nn.ReLU()

        self.linear_relu_stack = nn.Sequential(
            self.input_layer,
            self.relu,
            self.hidden_layer,
            self.relu,
            self.output_layer,
            self.relu,
        )

    def forward(self, history_vec):
        x = self.linear_relu_stack(history_vec)
        return x


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

    def forward(self, concat_vec):
        x = self.flatten(concat_vec)
        x = self.linear_relu_stack(x)
        return x


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

    def forward(self, next_states, back_context_out):
        x = torch.cat((next_states, back_context_out), dim=-1)
        return self.linear_relu_stack(x)



if __name__ == "__main__":
    net = Backbone()
    print(net.forward(torch.zeros((1, 128)), torch.zeros((1, 1))).shape)

    net = ContextEncoder()
    print(net.forward(torch.zeros((1, 129 * 16))).shape)

    net = Head(128, 128, 128)
    print(net.forward(torch.zeros((1, 128))).shape)

    net = RewardModel(128, 128, 128)
    print(net.forward(torch.zeros((1, 128)), torch.zeros((1, 128))).shape)
