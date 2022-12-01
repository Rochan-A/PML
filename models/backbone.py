import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self, hidden_layer_size=256, state_size=128, backbone_nw_op_size=256):
        super(Backbone, self).__init__()

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(state_size + 1, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, backbone_nw_op_size)
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


if __name__ == "__main__":
    net = Backbone()
    print(net.forward(torch.zeros((1, 128)), torch.zeros((1, 1))).shape)
