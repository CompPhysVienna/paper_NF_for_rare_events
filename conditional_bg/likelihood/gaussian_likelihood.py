import numpy as np
import torch

class gaussian_likelihood(object):
    
    def __init__(self, k, center=None, period=None):
        
        self.k = k
        self.center = center
        self.period = period # Half the periodic length
    
    def bias(self, cv, ref):
        
        if ref is None:
            ref = self.center
        
        ref = np.ones(cv.shape) * ref
        
        diff = cv-ref
        if self.period:

            diff = np.where(diff >  self.period/2, diff - self.period, diff)
            diff = np.where(diff < -self.period/2, diff + self.period, diff)
            
        return self.k/2 * (diff)**2
    
    
    def log_prob(self, generated_condition, reference_condition):
        
        if reference_condition == None:
            reference_condition = self.center
        
        cond_diff = generated_condition - reference_condition
        
        if self.period:

            cond_diff = torch.where(cond_diff >  self.period/2, cond_diff - self.period, cond_diff)
            cond_diff = torch.where(cond_diff < -self.period/2, cond_diff + self.period, cond_diff)
        
        prb = self.k/2 * torch.square( cond_diff )
        
        return prb
    
    
    def parameter_dict(self):
    
        properties = {}
        
        properties["Force Constant"] = self.k
        properties["Default Center"] = self.center
        properties["Period"] = self.period
         
        return properties