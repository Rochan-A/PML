from torch import nn

class backbone_nw(nn.Module):
    def __init__(self, hidden_layer_size=256, state_size=128, backbone_nw_op_size=256):
        super(backbone_nw, self).__init__()

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(state_size+1, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, backbone_nw_op_size)
        self.relu = nn.ReLU()

        self.linear_relu_stack = nn.Sequential(
            self.input_layer,
            self.relu, 
            self.hidden_layer,
            self.relu, 
            self.output_layer,
            self.relu
        )
    
    def forward(self, state_vector_t, action_t, num_hidden_layers=-1):
        x = torch.cat((state_vector_t, action_t), dim=0)
        output = self.linear_relu_stack(x)
        return output