from torch import nn


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


if __name__ == "__main__":
    import torch

    net = ContextEncoder()
    print(net.forward(torch.zeros((1, 129 * 16))).shape)
