from torch import nn

class head(torch.nn):
    def __init__(self, concat_vec_size, hidden_layer_size, state_size):
        super(head, self).__init__()

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(concat_vec_size, hidden_layer_size)
        self.hidden_layer = nn.linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, state_size)
        self.relu = nn.ReLU

        self.linear_relu_stack = nn.Sequential(
            self.input_layer,
            self.relu, 
            self.hidden_layer,
            self.relu, 
            self.output_layer,
            self.relu
        )
    
    def forward(concat_vec, num_hidden_layers):
        x = self.flatten(concat_vec)
        x = self.linear_relu_stack(x)
        return x[:-1], x[-1]

