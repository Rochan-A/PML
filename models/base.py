import torch
from torch import nn
import pathlib

class Backbone(nn.Module):
    def __init__(self, state_sz, action_sz, hidden_dim=256, hidden_layers=1, out_dim=256, actv=nn.LeakyReLU):
        super(Backbone, self).__init__()

        self.in_dim = state_sz + action_sz
        self.input_layer = nn.Linear(self.in_dim, hidden_dim)
        hiddens = []
        for _ in range(hidden_layers):
            hiddens.extend([
                nn.Linear(hidden_dim, hidden_dim),
                actv()
                ])
        self.hidden_layers = nn.ModuleList(hiddens)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, s_t, a_t):
        x = torch.cat((s_t, a_t), dim=-1)
        assert x.shape[-1] == self.in_dim, "Error with input, got shape {}, expected {}".format(x.shape, self.in_dim)
        x = self.input_layer(x)
        for _, l in enumerate(self.hidden_layers):
            x = l(x)
        return self.output_layer(x)

    def save(self, save_dir):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / 'backbone.pth')

    def load(self, load_dir):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / 'backbone.pth')
        self.load_state_dict(model_dict["state_dict"])



class ContextEncoder(nn.Module):
    def __init__(
        self, state_sz, action_sz, history_size=16, hidden_dim=256, hidden_layers=1, out_dim=256, actv=nn.LeakyReLU, deterministic=True, no_context=False
        ):
        super(ContextEncoder, self).__init__()

        self.in_dim = (state_sz + action_sz) * history_size
        self.deterministic = deterministic
        self.input_layer = nn.Linear(self.in_dim, hidden_dim)
        hiddens = []
        for _ in range(hidden_layers):
            hiddens.extend([
                nn.Linear(hidden_dim, hidden_dim),
                actv()
                ])
        self.hidden_layers = nn.ModuleList(hiddens)
        if self.deterministic:
            self.output_layer = nn.Linear(hidden_dim, out_dim)
        else:
            self.mu = nn.Linear(hidden_dim, out_dim)
            self.log_var = nn.Linear(hidden_dim, out_dim)

    def reparameterize(self,
                       mu,
                       log_var):
        """
        :param mu: (Tensor) mean of the latent Gaussian  [B x D]
        :param log_var: (Tensor) Log variance of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """

        std = torch.exp(0.5 * log_var)
        e = torch.randn_like(std)
        z = e * std + mu
        return z

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, "Error with input, got shape {}, expected {}".format(x.shape, self.in_dim)
        x = self.input_layer(x)
        for _, l in enumerate(self.hidden_layers):
            x = l(x)
        if self.deterministic:
            return self.output_layer(x)
        else:
            mu, log_var = self.mu(x), self.log_var(x)
            return self.reparameterize(mu, log_var)

    def save(self, save_dir):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / 'context.pth')

    def load(self, load_dir):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / 'context.pth')
        self.load_state_dict(model_dict["state_dict"])


if __name__ == "__main__":
    net = Backbone(128, 1)
    print(net.forward(torch.zeros((1, 128)), torch.zeros((1, 1))).shape)

    net = ContextEncoder(128, 1)
    print(net.forward(torch.zeros((1, 129 * 16))).shape)
