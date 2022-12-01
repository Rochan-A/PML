import numpy as np
import torch
import torch.nn as nn

from .base import ContextEncoder, Head, Backbone, RewardModel

def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

def CPU(var):
    return var.cpu().detach()


class DM(nn.Module):
    def __init__(self, env, config):
        super(DM).__init__()

        state_sz = env.observation_space.shape[0]

        self.context = ContextEncoder(config.context.history_sz, state_sz, config.context.hidden_sizes, config.context.out_dim)
        self.backbone = Backbone(config.backbone.hidden_sizes, state_sz, config.backbone.out_dim)
        self.reward = RewardModel(state_sz, config.reward.model.hidden_sizes, config.head.ensemble_size, config.backbone.out_dim)
        self.heads = [Head() for _ in range(config.head.ensemble_size)]

    def forward(self, state, action, history=None):
        raise NotImplementedError

    def reward(self, state, action, next_state, history=None):
        raise NotImplementedError

    def context(self, history):
        raise NotImplementedError


class DynamicsModel():
    def __init__(self, env, config):
        super().__init__()

        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.shape
        self.input_dim = self.state_dim + self.action_dim

        self.n_epochs = config.dynamics.n_epochs
        self.lr = config.dynamics.learning_rate
        self.batch_size = config.dynamics.batch_size
        
        self.save_model_flag = config.dynamics.save_model_flag
        self.save_model_path = config.dynamics.save_model_path
        
        self.validation_flag = config.dynamics.validation_flag
        self.validate_freq = config.dynamics.validation_freq
        self.validation_ratio = config.dynamics.validation_ratio

        if config.dynamics.load_model:
            self.model = torch.load(config.head["model_path"])
        else:
            self.model = DM(env, config)

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.dataset = []

    def process_dataset(self, dataset):
        # dataset format: list of [task_idx, state, action, next_state-state]
        data_list = []
        for data in dataset:
            s = data[1] # state
            a = data[2] # action
            label = data[3] # here label means the (next state - state) [state dim]
            data = np.concatenate((s, a), axis=0) # [state dim + action dim]
            data_torch = CUDA(torch.Tensor(data))
            label_torch = CUDA(torch.Tensor(label))
            data_list.append([data_torch, label_torch])
        return data_list

    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        with torch.no_grad():
            delta_state = self.model(inputs)
            delta_state = CPU(delta_state).numpy()
        return delta_state
    
    def add_data_point(self, data):
        # data format: [task_idx, state, action, next_state-state]
        self.dataset.append(data)
        
    def reset_dataset(self, new_dataset = None):
        # dataset format: list of [task_idx, state, action, next_state-state]
        if new_dataset is not None:
            self.dataset = new_dataset
        else:
            self.dataset = []
            
    def make_dataset(self, dataset, make_test_set = False):
        # dataset format: list of [task_idx, state, action, next_state-state]
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
            for datas, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(datas)
                loss = self.criterion(outputs, labels)
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
        for datas, labels in testloader:
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
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