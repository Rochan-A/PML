from torch import nn


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


if __name__ == "__main__":
    import torch

    net = Head(128, 128, 128)
    print(net.forward(torch.zeros((1, 128))).shape)
