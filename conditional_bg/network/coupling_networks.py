import torch
from torch import nn
import numpy as np


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, condition_dim, n_hidden, n_layers, init_zeros=True, activation=nn.ReLU):
        super(MLP, self).__init__()
        
        self.conditioned = False
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.init_zeros = init_zeros
        self.activation = activation
            
        if condition_dim != None:
            self.conditioned = True
            self.in_dim = in_dim + condition_dim
        
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden for _ in range(n_layers-1)]
        
        modules = []
        
        modules.append(nn.Linear(self.in_dim, n_hidden[0]))
        
        for i in range(1, n_layers-1):
            modules.append(activation())
            modules.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
            
        modules.append(activation())
        modules.append(nn.Linear(n_hidden[-1], self.out_dim))
        
        if init_zeros:
            modules[-1].weight.data.fill_(0)
            modules[-1].bias.data.fill_(0)
        
        self.net = nn.Sequential(*modules)

    
    def parameter_dict(self):
    
        properties = {}
        
        properties["N Hidden"] = self.n_hidden
        properties["N Layers"] = self.n_layers
        properties["Initialize Zeros"] = self.init_zeros
        
        return properties
    
        
    def forward(self, x, condition = None):
        
        if self.conditioned:
            x_condition = torch.cat((x, condition), 1)
            return self.net(x_condition)
        else:
            return self.net(x)
        
        
