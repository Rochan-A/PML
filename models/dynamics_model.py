import numpy as np
import torch
import torch.nn as nn

from base import ContextEncoder, Head, Backbone, RewardModel

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


def compute_head_loss(outputs, labels, criterion):
    raise NotImplementedError


def select_head(outputs, labels, criterion):
    raise NotImplementedError


class DynamicsModel():
    def __init__(self, env, config):
        super().__init__()

        self.device = config.device

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = len(env.action_space.shape)

        self.n_epochs = config.dynamics.n_epochs
        self.lr = config.dynamics.learning_rate
        self.batch_size = config.dynamics.batch_size

        self.save_model_flag = config.dynamics.save_model_flag
        self.save_model_path = config.dynamics.save_model_path

        self.validation_flag = config.dynamics.validation_flag
        self.validate_freq = config.dynamics.validation_freq
        self.validation_ratio = config.dynamics.validation_ratio

        if config.dynamics.load_model:
            self.model = torch.load(config.head["model_path"]).to(device=self.device)
        else:
            self.model = DNet(env, config).to(device=self.device)

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.dataset = []

    def process_dataset(self, dataset):
        # dataset format: list of [history, state, action, next_state-state]
        data_list = []
        for data in dataset:
            hist = data[0] # history
            s = data[1] # state
            a = data[2] # action
            label = data[3] # here label means the (next state - state) [state dim]
            hist_tensor = torch.Tensor(hist).to(dtype=torch.float32, device=self.device)
            state_tensor = torch.Tensor(s).to(dtype=torch.float32, device=self.device)
            action_tensor = torch.Tensor(a).to(dtype=torch.float32, device=self.device)
            label_tensor = torch.Tensor(label).to(dtype=torch.float32, device=self.device)
            data_list.append([hist_tensor, state_tensor, action_tensor, label_tensor])
        return data_list

    def predict(self, s, a, hist):
        s = torch.tensor(s).to(dtype=torch.float32, device=self.device)
        a = torch.tensor(a).to(dtype=torch.float32, device=self.device)
        hist = torch.tensor(hist).to(dtype=torch.float32, device=self.device)
        with torch.no_grad():
            bb_embb, c_embb, delta_states = self.model.forward(s, a, hist)
            bb_embb = bb_embb.cpu().numpy()
            c_embb = c_embb.cpu().numpy()
            for idx in range(len(delta_states)):
                delta_states[idx] = delta_states[idx].cpu().numpy()
        return bb_embb, c_embb, delta_states

    def add_data_point(self, data):
        # data format: [history, state, action, next_state-state]
        self.dataset.append(data)

    def reset_dataset(self, new_dataset = None):
        # dataset format: list of [history, state, action, next_state-state]
        if new_dataset is not None:
            self.dataset = new_dataset
        else:
            self.dataset = []

    def make_dataset(self, dataset, make_test_set = False):
        # dataset format: list of [history, state, action, next_state-state]
        num_data = len(dataset)
        data_list = self.process_dataset(dataset)

        if make_test_set:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]
            train_set = [data_list[idx] for idx in train_idx]
            test_set = [data_list[idx] for idx in test_idx]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
            if len(test_set):
                test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_loader = torch.utils.data.DataLoader(data_list, shuffle=True, batch_size=self.batch_size)
            test_loader = None
        return train_loader, test_loader

    def fit(self, dataset=None, logger = True):
        if dataset is not None:
            train_loader, test_loader = self.make_dataset(dataset, make_test_set=self.validation_flag)
        else: # use its own accumulated data
            train_loader, test_loader = self.make_dataset(self.dataset, make_test_set=self.validation_flag)

        for epoch in range(self.n_epochs):
            loss_this_epoch = []
            for hists, states, actions, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model.forward(states, actions, hists)
                loss = compute_head_loss(outputs, labels, self.criterion)
                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())

            if self.save_model_flag:
                torch.save(self.model, self.save_model_path)

            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                loss_test = 11111111
                if test_loader is not None:
                    loss_test = self.validate_model(test_loader)
                loss_train = self.validate_model(train_loader)
                if logger:
                    print(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss_train:.4f}, loss test  {loss_test:.4f}")

        return np.mean(loss_this_epoch)

    def validate_model(self, testloader):
        loss_list = []
        for hists, states, actions, labels in testloader:
            outputs = self.model(states, actions, hists)
            loss = compute_head_loss(outputs, labels, self.criterion)
            loss_list.append(loss.item())
        return np.mean(loss_list)

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__=='__main__':
    import gym
    import sunblaze_envs

    import yaml
    from easydict import EasyDict

    with open('./configs/cartpole.yaml') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    env = sunblaze_envs.make(config.env)
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