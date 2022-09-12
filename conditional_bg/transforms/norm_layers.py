
import torch
from conditional_bg.transforms.base import base_transform

class norm_layer(base_transform):
    
    def __init__(self, train_data, device):
        super().__init__(device)
        
        self.means = torch.nn.Parameter(train_data.mean(axis=0, keepdim=True), requires_grad=False)
        self.standard_deviations =  torch.nn.Parameter(train_data.std(axis=0, keepdim=True), requires_grad=False)
    
    def F_xz(self, x, return_Jacobian=True):
        
        z = (x -  self.means) / self.standard_deviations 
        
        if return_Jacobian:
            log_det_xz = (-torch.log(self.standard_deviations).repeat(x.shape[0], 1)).sum(dim=-1, keepdims=True)
        else:
            log_det_xz = None
            
        return z, log_det_xz
    
    
    def F_zx(self, z, return_Jacobian=True):
            
        x = z * self.standard_deviations + self.means
        
        if return_Jacobian:
            log_det_zx = torch.log(self.standard_deviations).repeat(x.shape[0], 1).sum(dim=-1, keepdims=True)
        else:
            log_det_zx = None
            
        return x, log_det_zx
    