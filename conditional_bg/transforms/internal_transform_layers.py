import numpy as np

import torch

from conditional_bg.transforms.base import base_transform

        
class internal_transformation_2D(base_transform):
    
    def __init__(self, n_particles, angle_shift, device, jacobian_regularization = 0, first_angle_unsigned = False):
        super().__init__(device)
        
        self.n_particles = n_particles
        self.dofs = n_particles * 2
        
        angle_mask_np = np.zeros(n_particles * 2 - 3).astype(np.float32)
        angle_mask_np[1::2] = 1
        
        self.angle_mask = torch.from_numpy(angle_mask_np).to(self.device)
        
        self.angle_shift = torch.nn.Parameter(angle_shift.new_zeros(n_particles*2 - 3).unsqueeze(0), requires_grad=False)
        
        self.angle_shift[:, 1::2] = angle_shift
        
        self.jacobian_regularization = jacobian_regularization
        self.first_angle_unsigned = first_angle_unsigned
        
        self.delta_dofs = -3
        self.in_torsion_loss = True
        
        
    def angle(self, v, w):
        
        b = v[:, 0]*w[:, 0] + v[:, 1]*w[:, 1]   
        a = v[:, 0]*w[:, 1] - v[:, 1]*w[:, 0]

        return -torch.atan2(a, b)
        
        
    def rotate_v(self, v, theta):
        
        v_r = torch.zeros_like(v)
        
        v_r[:, 0] = v[:, 0] * torch.cos(theta) + v[:, 1] * torch.sin(theta)
        v_r[:, 1] = -v[:, 0] * torch.sin(theta) + v[:, 1] * torch.cos(theta)
        
        return v_r


    def to_internal(self, x0, x1, x2):

        # x0.shape = [B x P x 2]

        v_10 = x0 - x1

        r_ij = torch.linalg.norm(v_10, dim=-1, keepdims=True)


        v_10 = x0 - x1
        v_12 = x2 - x1

        b = v_10[:, :, 0]*v_12[:, :, 0] + v_10[:, :, 1]*v_12[:, :, 1]   
        a = v_10[:, :, 0]*v_12[:, :, 1] - v_10[:, :, 1]*v_12[:, :, 0]

        theta_ijk = -torch.atan2(a, b).unsqueeze(-1)

        logdet_xz = -torch.log(r_ij[:, 1:-1] + self.jacobian_regularization).sum(dim=1).reshape(-1, 1)
        
        internal = torch.cat([r_ij, theta_ijk], dim=-1).reshape(x0.shape[0], -1)[:, :self.dofs-3]

        internal = internal - np.pi * self.angle_mask + self.angle_shift
        internal = torch.where(self.angle_mask * internal > -np.pi, internal, internal + 2*np.pi)

        
        return internal, logdet_xz
    
    
    def to_cartesian(self, internal):
            
        z = torch.zeros_like(internal)
        z[:, 1::2] = internal[:, 1::2]
        z[:, ::2] = torch.abs(internal[:, ::2])
        
        logdet_zx = torch.log(z[:, ::2] + self.jacobian_regularization)[:, 1:].sum(dim=1).reshape(-1, 1)
        
        z = z + self.angle_mask * np.pi - self.angle_shift
        z = torch.where(self.angle_mask * z < np.pi, z, z - 2*np.pi)
            
        transform = z.new_zeros([z.shape[0], self.n_particles, 2])
        
        # 1st particle arbitrary at (0,0)
        # 2nd particle at (0, d_12)
        transform[:, 1, 0] = z[:, 0]
        
        # 3rd particle and onwards
        for i in range(2, self.n_particles):
            
            angle_index = 1 + 2 * (i-2)
            d_index = 2 + 2 * (i-2)
            
            transform[:, i] = transform[:, i-1] + self.rotate_v(transform[:, i-2]-transform[:, i-1], z[:, angle_index]) * (z[:, d_index]/z[:, d_index-2]).unsqueeze(1)
            
            
        x = transform.reshape(z.shape[0], self.n_particles*2)
        
        return x, logdet_zx
    
    
    def get_angles(self, internal, norm=True):
            
        angles = internal * self.angle_mask.unsqueeze(0)
        
        if self.first_angle_unsigned:
            angles[:, 1] = angles[:, 1] * 2 - np.pi
        
        return angles
    
        
    def F_xz(self, x, return_Jacobian=True):
        
        x_p =  x.reshape(x.shape[0], self.n_particles, 2)

        x0 = x_p[:, np.roll(np.arange(self.n_particles), 0)]
        x1 = x_p[:, np.roll(np.arange(self.n_particles), -1)]
        x2 = x_p[:, np.roll(np.arange(self.n_particles), -2)]

        z, logdet_xz = self.to_internal(x0, x1, x2)
        
        return z, logdet_xz
    
    
    def F_zx(self, z, return_Jacobian=True):
        
        x, logdet_zx = self.to_cartesian(z)
            
        return x, logdet_zx