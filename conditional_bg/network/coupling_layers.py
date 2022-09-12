import numpy as np

import torch
from torch import nn

from conditional_bg import util


class affine_coupling_layer(nn.Module):
    
    def __init__(self, dimension, masks, transform_network, transform_network_parameters, 
                 condition_dim, clamp_scale=None, scale_tanh=False):
        super(affine_coupling_layer, self).__init__()
        
        self.masks = nn.Parameter(masks, requires_grad=False)
        
        for m in self.masks:
            assert m.shape[0] == dimension, "Mask of length {} does not match the input dimension of {}!".format(m.shape[0], dimension)
        
        self.blocks = len(self.masks)
        self.clamp_scale = clamp_scale
        self.scale_tanh = scale_tanh
        
        self.transform_network = transform_network
        
        self.n_parameters = 1 if self.scale_tanh else 2
        
        transform_layers, scale_layers = [], []
        
        self.split_indices = []
        
        for i in range(self.blocks):
            
            identity_index = torch.nonzero(self.masks[i] == 0, as_tuple=True)[0]
            transform_index = torch.nonzero(1-self.masks[i] == 0, as_tuple=True)[0]
            
            identity_dim = torch.count_nonzero(self.masks[i] == 0)
            transform_dim = torch.count_nonzero(1 - self.masks[i] == 0)

            transform_layers.append(transform_network(identity_dim, transform_dim * self.n_parameters, condition_dim, **transform_network_parameters))
            
            if self.scale_tanh:
                scale_network_parameters = transform_network_parameters.copy()
                scale_network_parameters["activation"] = nn.Tanh
                
                scale_layers.append(transform_network(identity_dim, transform_dim * self.n_parameters, condition_dim, **scale_network_parameters))
            
            self.split_indices.append([identity_index, transform_index])
        
        if self.scale_tanh:
            self.s = nn.ModuleList(scale_layers)
            
        self.t = nn.ModuleList(transform_layers)
        
        self.T_dependent = False
        
        
    def parameter_dict(self):
    
        properties = {}
        
        mask_string = util.mask_parser(self.masks)
        
        properties["Masks"] = mask_string
        properties["Network"] = self.transform_network.__class__.__name__
        properties["Scale Clamping"] = self.clamp_scale
        properties["Scale Netowrk Tanh"] = self.scale_tanh
        
        return properties
        
    
    def F_zx(self, z, condition = None):
        
        log_det_Jzx, x = z.new_zeros(z.shape[0], 1), z
        
        for i in range(self.blocks):
            
            identity_index, transform_index = self.split_indices[i]
            
            if not self.scale_tanh:
                transform_output = self.t[i](x[:,identity_index], condition).reshape(x.shape[0], -1, self.n_parameters)
                
                S_init = transform_output[:, :, 0]
                T =  transform_output[:, :, 1]
            
            else:
                T = self.t[i](x[:,identity_index], condition)
                S_init = self.s[i](x[:,identity_index], condition)
                
            
            S = S_init
            
            if self.clamp_scale is not None:
                S_init = self.clamp_scale * 2/np.pi * torch.atan(S_init)
            
            x[:,transform_index] = x[:,transform_index] * torch.exp(S) + T
            
            log_det_Jzx += torch.sum(S , dim=1, keepdims=True)
            
        return x, log_det_Jzx

    
    def F_xz(self, x, condition = None):
       
        log_det_Jxz, z = x.new_zeros(x.shape[0], 1), x
        
        for i in reversed(range(self.blocks)):
            
            identity_index, transform_index = self.split_indices[i]
            
            if not self.scale_tanh:
                transform_output = self.t[i](z[:,identity_index], condition).reshape(z.shape[0], -1, self.n_parameters)
                
                S_init = transform_output[:, :, 0]
                T =  transform_output[:, :, 1]
            
            else:
                T = self.t[i](x[:,identity_index], condition)
                S_init = self.s[i](x[:,identity_index], condition)
                
            
            if self.clamp_scale is not None:
                S_init = self.clamp_scale * 2/np.pi * torch.atan(S_init)
                
            S = S_init
                
            z[:, transform_index] = torch.exp(-S) * (z[:, transform_index] - T)

            log_det_Jxz -= torch.sum(S, dim=1, keepdims=True)
            
        return z, log_det_Jxz