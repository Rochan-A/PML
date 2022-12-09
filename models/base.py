import torch
from torch import nn
import pathlib
from torch.distributions import Normal


ACTV = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'identity': nn.Identity
}


class MLP(nn.Module):
    def __init__(self,
                 in_dim=1,
                 hidden_dim=200,
                 hidden_layers=1,
                 out_dim=32,
                 hidden_actv='relu',
                 output_actv='relu'
        ):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_layer = nn.Linear(self.in_dim, hidden_dim)
        hiddens = []
        for _ in range(hidden_layers):
            hiddens.extend([
                ACTV[hidden_actv](),
                nn.Linear(hidden_dim, hidden_dim)
                ])
        self.hidden_layers = nn.ModuleList(hiddens)
        self.output_layer = nn.Sequential(
            ACTV[hidden_actv](),
            nn.Linear(hidden_dim, out_dim),
            ACTV[output_actv]()
        )

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, \
            "Error with input, got shape {}, expected {}".format(x.shape, self.in_dim)
        x = self.input_layer(x)
        for _, l in enumerate(self.hidden_layers):
            x = l(x)
        return self.output_layer(x)


class StateActionEncoder(nn.Module):
    def __init__(self,
                 state_sz=5,
                 action_sz=1,
                 hidden_dim=200,
                 hidden_layers=1,
                 out_dim=256,
                 hidden_actv='relu',
                 state_actv='identity',
                 action_actv='tanh'
        ):
        super(StateActionEncoder, self).__init__()

        self.state_sz = state_sz
        self.action_sz = action_sz
        self.out_dim = out_dim
        self.in_dim = state_sz + action_sz

        self.state_enc = MLP(
            in_dim=state_sz,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            out_dim=int(out_dim/2),
            hidden_actv=hidden_actv,
            output_actv=state_actv
        )

        self.action_enc = MLP(
            in_dim=action_sz,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            out_dim=int(out_dim/2),
            hidden_actv=hidden_actv,
            output_actv=action_actv
        )

    def forward(self, s_t, a_t):
        assert s_t.shape[-1] == self.state_sz, \
            "Error with stateinput, got shape {}, expected {}".format(s_t.shape, self.state_sz)
        assert a_t.shape[-1] == self.action_sz, \
            "Error with action input, got shape {}, expected {}".format(a_t.shape, self.action_sz)
        return self.state_enc.forward(s_t), self.action_enc.forward(a_t)

    def joint_embb(self, s_t, a_t):
        s_embb, a_embb = self.forward(s_t, a_t)
        return torch.cat([s_embb, a_embb], dim=-1)

    def save(self, save_dir):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
        }
        torch.save(model_dict, pathlib.Path(save_dir) / 'state_action_enc.pth')

    def load(self, load_dir):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / 'state_action_enc.pth')
        self.load_state_dict(model_dict["state_dict"])


class ContextEncoder(nn.Module):
    def __init__(self,
                 state_sz=5,
                 action_sz=1,
                 history_size=50,
                 hidden_dim=200,
                 hidden_layers=1,
                 out_dim=16,
                 hidden_actv='relu',
                 output_actv='identity',
                 no_context=False
        ):
        super(ContextEncoder, self).__init__()

        self.in_dim = (state_sz + action_sz) * history_size
        self.out_dim = out_dim
        self.no_context = no_context
        self.input_layer = nn.Linear(self.in_dim, hidden_dim)
        hiddens = []
        for _ in range(hidden_layers):
            hiddens.extend([
                ACTV[hidden_actv](),
                nn.Linear(hidden_dim, hidden_dim),
                ])
        hiddens.append(ACTV[hidden_actv]())
        self.hidden_layers = nn.ModuleList(hiddens)
        if self.no_context:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                ACTV[output_actv]()
            )
        else:
            self.mu = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                ACTV[output_actv]()
            )
            self.log_var = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                ACTV[output_actv]()
            )

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, \
            "Error with input, got shape {}, expected {}".format(x.shape, self.in_dim)
        x = self.input_layer(x)
        for _, l in enumerate(self.hidden_layers):
            x = l(x)
        if self.no_context:
            return self.output_layer(x)
        else:
            mu, log_var = self.mu(x), self.log_var(x)
            return self.reparameterize(mu, log_var), mu, log_var

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

    def dist(self,
             x):
        mu, log_var = self.forward(x)
        return Normal(mu, log_var)

    def save(self, save_dir):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
        }
        torch.save(model_dict, pathlib.Path(save_dir) / 'context.pth')

    def load(self, load_dir):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / 'context.pth')
        self.load_state_dict(model_dict["state_dict"])


if __name__ == "__main__":
    net = StateActionEncoder()
    print(net.joint_embb(torch.zeros((1, 5)), torch.zeros((1, 1))).shape)

    net = ContextEncoder()
    print(net.reparameterize(torch.zeros((1, 6 * 50))).shape)
