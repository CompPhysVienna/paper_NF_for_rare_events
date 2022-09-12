
from torch import nn

class base_model(nn.Module):
    
    def __init__(self, n_particles, dimensions):
        super().__init__() 
        
        self.dofs = n_particles * dimensions
        
        self.n_particles = n_particles
        self.dimensions = dimensions
        
        self.multi_T_supported = False
        
    
    def get_model_parameters(self):
        
        return NotImplementedError
    

    def initial_configuration(self, params):
        
        raise NotImplementedError
        
    
    def potential_energy(x, params):
        
        raise NotImplementedError

    
    def _force(x, params):
       
        raise NotImplementedError
    
    
    def log_prob(self, x):

        raise NotImplementedError
        
    
    def sample(self, x, T = None, generator=None):

        raise NotImplementedError

        
    def sample_with_bias(self, x):

        raise NotImplementedError

    
    def loss_PBC(self, x):
    
        raise NotImplementedError    
    
    
    def parameter_dict(self):
    
        properties = self.additional_parameters()
        
        properties["DOFs"] = self.dofs
        properties["N Particles"] = self.n_particles
        properties["Dimensions"] = self.dimensions
        properties["Multi-T Supported"] = self.multi_T_supported
         
        return properties
    
        
    def additional_parameters(self):
    
        return {}