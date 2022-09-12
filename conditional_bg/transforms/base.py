
from torch import nn

class base_transform(nn.Module):
    
    def __init__(self, device):
        super(base_transform, self).__init__()
        
        self.device = device
        
        self.delta_dofs = 0
        self.in_torsion_loss = False
    
    def F_xz(self):
        raise NotImplementedError
    
    def F_zx(self):
        
        raise NotImplementedError

    