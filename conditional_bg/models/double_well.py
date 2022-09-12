import numpy as np

import torch
from torch import nn
from numba import jit

from conditional_bg.models.base import base_model
from conditional_bg.samplers.biased_metropolis_monte_carlo import biased_metropolis_monte_carlo

class double_well_model(base_model):
    
    def __init__(self, device, B = 3):
        super().__init__(n_particles = 1, dimensions = 2) 
        
        self.B = B
        
        self.device = device
        self.MC = biased_metropolis_monte_carlo(self)
    
    
    def get_model_parameters(self):
        
        return np.array([self.B])
    
    
    def potential_energy(self, x):
    
        return self.B * ((x[0]**2 - 1)**2 + (x[0]-x[1])**2)

    
    def potential_energy_batch(self, x):
    
        return self.B * ((x[:,0]**2 - 1)**2 + (x[:,0]-x[:,1])**2)

    
    def initial_configuration(self, L=1):
        
        return L*(np.random.random(size=self.dofs)*2 - 1)
    
        
    @staticmethod
    @jit(nopython=True) 
    def _force(x, model_parameters):
        
        F = np.zeros(2, dtype=np.float32)
        
        F[0] = 2* model_parameters[0] * (2 * x[0]**3 -x[0] - x[1])

        F[1] = -2* model_parameters[0] * (x[0] - x[1])

        return -F
    
    
    def force(self, x):
        
        return self._force(x, self.get_model_parameters())
    
    
    def force_torch(self, x):
        
        F = x.new_zeros((x.shape[0], 2))
        
        F[:, 0] = 2 * self.B * (2 * x[:, 0]**3 -x[:, 0] - x[:, 1])

        F[:, 1] = -2* self.B * (x[:, 0] - x[:, 1])

        return -F
    
    
    def sample(self, n_samples, T=1, n_cycles=100, step=.5, cv=lambda x:x, likelihood=None, ref_center=None, shuffle=False, generator=None):
        
        sample = self.MC.sample_space(n_samples, T, n_cycles, step, cv, likelihood, ref_center, shuffle)
        
        return torch.from_numpy(sample[0].astype(np.float32)).to(self.device)

        
    def log_prob(self, x):
    
        return (self.B * ((x[:,0]**2 - 1)**2 + (x[:,0]-x[:,1])**2)).unsqueeze(-1)
    

    def additional_parameters(self):

        return {"B" : self.B}