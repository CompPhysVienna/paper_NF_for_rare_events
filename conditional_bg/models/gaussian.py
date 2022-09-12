import numpy as np

import torch
from torch import nn
from torch import distributions

from conditional_bg.models.base import base_model
    
    
class gaussian_model(base_model):
    
    def __init__(self, n_particles, dimensions, device, reference_T = 1):
        super().__init__(n_particles = n_particles, dimensions = dimensions) 
        
        self.reference_T = reference_T
        
        self.sd = 1
        
        self.device = device
        
        self.multi_T_supported = True
        
        
    def sample(self, N, T = None, generator=None):
        
        if T is None:
            # Assume one wants to sample at reference temperature
            T = torch.ones((N, 1), dtype = torch.float32, device = self.device) * self.reference_T

            
        if isinstance(T, (torch.Tensor)):

            assert T.shape == (N, 1), "Temperature tensor should match dimensions [N x 1], where N is the number of samples to generate."

            T_sample = T / self.reference_T

        elif isinstance(T, (float, int)):

            T_sample = torch.ones((N, 1), dtype = torch.float32, device = self.device) * T / self.reference_T
            
            
        g = torch.empty([N, self.dofs], dtype = torch.float32, device = self.device).normal_(mean=0, std=self.sd, generator=generator)
        
        return g * torch.sqrt(T_sample)
       
        
    def log_prob(self, z):
        
        prb = (self.reference_T) * 1/(2*(self.sd**2)) * z.pow(2).sum(dim=-1, keepdims=True)
        
        return prb
    
    
    def additional_parameters(self):

        return {"Reference Temperature" : self.reference_T, "SD" : self.sd}
    
    