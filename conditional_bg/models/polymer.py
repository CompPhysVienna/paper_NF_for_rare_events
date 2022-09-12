import numpy as np

import torch
from torch import nn

from numba import jit

from conditional_bg.models.base import base_model

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import colorConverter

class polymer_model(base_model):
    
    def __init__(self, n_particles, dimensions, eps, sigma, d_bond, k_bond, theta_angle, k_angle, enhanced_sample_moves=True, distance_eps=1e-5):
        super().__init__(n_particles = n_particles, dimensions = dimensions) 
        
        self.eps = eps
        self.sigma = sigma
        self.d_bond = d_bond
        self.k_bond = k_bond
        self.theta_angle = theta_angle
        self.k_angle = k_angle
        self.distance_eps = distance_eps
        self.enhanced_sample_moves = enhanced_sample_moves
    
    def get_model_parameters(self):
        return np.array([self.n_particles,  self.dimensions, self.eps, self.sigma, self.d_bond, self.k_bond, self.theta_angle, self.k_angle])
    
    
    @staticmethod
    @jit(nopython=True) 
    def angle(v, w):
        
        b = v[:, 0]*w[:, 0] + v[:, 1]*w[:, 1]   
        a = v[:, 0]*w[:, 1] - v[:, 1]*w[:, 0]

        return -np.atan2(a, b)
    
    
    @staticmethod
    @jit(nopython=True) 
    def potential_energy(x, params):
        U_total = 0
        conf = x.reshape(int(params[0]), int(params[1]))
        
        for i in range(params[0]):
            for j in range(i+1, params[0]):
                if i != j:
                    r = np.linalg.norm(conf[j]-conf[i])
                    
                    term1 = np.power(params[3]/r, 6)
                    term2 = np.square(term1)
                    U_part = 4 * params[2] * (term2 - term1 )
                    U_total +=  U_part
        
        v_ij = conf[1:] - conf[:-1]
        
        r_ij = np.sqrt(v_ij[:,0]**2 + v_ij[:,1]**2)
        
        U_bonded = np.sum( params[5]/2 * np.square(r_ij - params[4]))
        
        U_total += U_bonded
        
        v = conf[:-2] - conf[1:-1]
        w = conf[2:] - conf[1:-1]
        
        b = v[:, 0]*w[:, 0] + v[:, 1]*w[:, 1]   
        a = v[:, 0]*w[:, 1] - v[:, 1]*w[:, 0]

        theta_ijk = -np.arctan2(a, b)
        
        U_angle = np.sum(params[7]/2 * (1 - np.cos(theta_ijk-params[6])))
        
        U_total += U_angle
                               
        return U_total

    
    @staticmethod
    @jit(nopython=True) 
    def _force(x, params):
        F = np.zeros((int(params[0]), int(params[1])), dtype=np.float32)
        
        conf = x.reshape(int(params[0]), int(params[1]))
        
        for i in range(params[0]):
            for j in range(i + 1, params[0]):
                r = np.linalg.norm(conf[i]-conf[j])
                
                u = (conf[i]-conf[j])/r

                term1 = np.power(params[3]/r, 6)
                term2 = np.square(term1)

                F_part = ((24 * params[2])/r) * ( 2*term2 - term1 ) * u

                F[i] +=  F_part
                F[j] -=  F_part
                
                
        v_ij = conf[1:] - conf[:-1]
        
        r_ij = np.sqrt(v_ij[:,0]**2 + v_ij[:,1]**2)
        
        F_bonded = params[5] * (r_ij - params[4])
        
        for i in range(params[0]-1):
            F_ij = F_bonded[i] * (v_ij[i]/r_ij[i])
            
            F[i] += F_ij
            F[i+1] += -F_ij
        
        
        v_ij = conf[:-2] - conf[1:-1]
        v_ik = conf[2:] - conf[1:-1]
        
        b = v_ij[:, 0]*v_ik[:, 0] + v_ij[:, 1]*v_ik[:, 1]   
        a = v_ij[:, 0]*v_ik[:, 1] - v_ij[:, 1]*v_ik[:, 0]

        theta_ijk = -np.arctan2(a, b)
        
        F_angle = params[7]/2 * np.sin(theta_ijk - params[6])
        
        v_angle = (conf[:-1] - conf[1:])[:,::-1]
        v_angle[:,1] *= -1
        v_angle_norm = np.sqrt(v_angle[:,0]**2 + v_angle[:,1]**2 ).reshape(-1, 1)
        v_angle /= v_angle_norm

        for i in range(1, params[0]-1):
            
            F_angle_ijk = F_angle[i-1]
            
            p_i = v_angle[i-1]
            p_k = v_angle[i]
            
            F_i = F_angle_ijk / r_ij[i-1] * p_i
            
            F_k = F_angle_ijk / r_ij[i] * p_k
            
            F_j = -F_i - F_k
            
            F[i-1] += F_i
            F[i] += F_j
            F[i+1] += F_k
        
        
        F = np.ravel(F)
        
        return F
    
    
    def log_prob(self, x):
        
        x = x.reshape((x.shape[0], self.n_particles, self.dimensions))

        rep_x = x.repeat(1,1,self.n_particles).reshape(x.shape[0], self.n_particles**2, self.dimensions)

        rep_x2 = x.repeat(1, self.n_particles,1)

        vec_ab = rep_x - rep_x2

        r_ab = torch.norm(vec_ab, dim=2)
        
    
        term1 = torch.pow(self.sigma/(r_ab + self.distance_eps), 6)

        term1[:, ::self.n_particles+1] = 0 

        term2 = torch.square(term1)

        U_part = 4 * self.eps * (term2 - term1 )

        U_total = 1/2 * U_part.sum(dim=1)

        
        v_ij = x[:, 1:] - x[:, :-1]
        
        r_ij = torch.sqrt(v_ij[:, :, 0]**2 + v_ij[:, :, 1]**2)
        
        U_bonded = torch.sum( self.k_bond/2 * torch.square(r_ij - self.d_bond), dim=-1)
        
        U_total += U_bonded
        
        
        v = x[:, :-2] - x[:, 1:-1]
        w = x[:, 2:] - x[:, 1:-1]
        
        b = v[:, :, 0]*w[:, :, 0] + v[:, :, 1]*w[:, :, 1]   
        a = v[:, :, 0]*w[:, :, 1] - v[:, :, 1]*w[:, :, 0]

        theta_ijk = -torch.atan2(a, b)
        
        U_angle = torch.sum(self.k_angle/2 * (1 - torch.cos(theta_ijk-self.theta_angle)), dim=-1)
        
        U_total += U_angle
        
        return U_total.unsqueeze(-1)
    
    
    def mirror(self, x):
        
        if np.random.random() < 0.1:
            x[::2] = -x[::2]
        
        return x
    
    def reroll_bonds(self, x):

        x_p = x.reshape(self.n_particles, self.dimensions)
        x_p = x_p[np.roll(np.arange(self.n_particles), np.random.randint(1, 7)), :]
        x = x_p.reshape(self.dofs)

        return x
    
    def reverse(self, x):
        
        if np.random.random() < 0.1:
            x_p = np.reshape(x, (self.n_particles, self.dimensions))[::-1].copy()
        
            x = np.reshape(x_p, (self.dofs))
            
        return x
    
    
    def MCMC(self, init_x, N_steps, step_size = 0.4, shuffle=False, kbT=1, stride=1):

        conf = []

        x = init_x
        acc, rej, rer = 0,0,0
        
        U_old = self.potential_energy(x, self.get_model_parameters())
        for i in range(N_steps):

            x_new = x.copy()
            dr = (np.random.random(self.dimensions) * 2 - 1) * step_size
            
            particle_index = np.random.randint(0, int(len(x_new)/2) )
            
            x_new[particle_index*2 : particle_index*2+2] += dr
            
            rerolled= False
            if np.random.random() < 0.9:
                
                x_new[particle_index*2 : particle_index*2+2] += dr
            
                x_new = self.reverse(x_new)
                x_new = self.mirror(x_new)
                
            else:
                rerolled = True
                x_new = self.reroll_bonds(x_new)
            
            x_avg = np.average(x_new[::2])
            y_avg = np.average(x_new[1::2])
            
            x_new[::2] -= x_avg
            x_new[1::2] -= y_avg
            
            U_new = self.potential_energy(x_new, self.get_model_parameters())

            if U_new < U_old:
                x = x_new.copy()
                U_old = U_new
                acc += 1
                if rerolled:
                    rer+=1
                
            elif np.random.random() < np.exp(-1/kbT * (U_new-U_old)):
                x = x_new.copy()
                U_old = U_new
                acc += 1
                if rerolled:
                    rer+=1
            else:
                rej += 1
            
            if (i+1) % stride == 0:
                conf.append(x)

        conf = np.array(conf)
        
        if shuffle:
            np.random.shuffle(conf)
        
        return conf, [acc,rej,rer]
    
    
    
    def bias_MCMC(self, init_x, N_steps, cv_function, ref_value, k, step_size = 0.4, kbT=1, stride=1):

        conf = []

        x = init_x
        acc = 0
        
        U_old = self.potential_energy(x, self.get_model_parameters()) + k/2 * (cv_function(x)-ref_value)**2
        
        for i in range(N_steps):
            
            x_new = x.copy()
            dr = (np.random.random(self.dimensions) * 2 - 1) * step_size
            
            particle_index = np.random.randint(0, int(len(x_new)/2) )
            
            if self.enhanced_sample_moves:
                
                if np.random.random() < (1 - (1/200)):

                    x_new[particle_index*2 : particle_index*2+2] += dr

                    x_new = self.reverse(x_new)
                    x_new = self.mirror(x_new)

                else:
                    x_new = self.reroll_bonds(x_new)
                    
            else:
                
                x_new[particle_index*2 : particle_index*2+2] += dr
                
                
            x_new[::2] -= x_new[::2].mean(axis=0)
            x_new[1::2] -= x_new[1::2].mean(axis=0)
            
            U_new = self.potential_energy(x_new, self.get_model_parameters()) +  k/2 * (cv_function(x_new)-ref_value)**2
            
            trial_acc = False
            
            if U_new < U_old:
                trial_acc = True
            elif np.random.random() < np.exp(1/kbT * (U_old-U_new)):
                trial_acc = True
            
            if trial_acc:
                x = x_new
                U_old = U_new
                acc += 1
                
            if (i+1) % stride == 0:
                conf.append(x)

        conf = np.array(conf)
        
        return conf, [acc, N_steps-acc]
    

    def plot_configuration(self, x, xlim=(-2,2), ylim=(-2,2), fig_size = (6 * 0.393701, 5 * 0.393701), permutation_inv = False):

        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i in range(len(color_list)):
            color_list[i] = colorConverter.to_rgba(color_list[i], alpha=0.75)

        fig, ax = plt.subplots(1, figsize = fig_size, dpi = 300)

        radius = (np.power(2,1/6) * self.sigma)/2
        particle_list = []

        configurations = x.reshape(self.n_particles, self.dimensions)

        for i, p in enumerate(configurations):
            color = color_list[i%10]
            
            if permutation_inv:
                color = (0.7,0.7,0.7)
                
            circle = plt.Circle(p, radius, facecolor=color, edgecolor=colorConverter.to_rgba((0,0,0), alpha=0.9), lw=1)
            a = ax.add_patch(circle)
            particle_list.append(a)

            if i < len(configurations) - 1:
                P1 = configurations[i]
                P2 = configurations[i+1]
                l, = ax.plot([P1[0], P2[0]], [P1[1], P2[1]], zorder=0, c="0", marker="o", markersize=2)
                
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.show()
    
    
    def visualize_trajectory(self, trajectory, smoothing=None, 
                             interval=500,
                             xlim=(-2,2), ylim=(-2,2), 
                             fig_size = (6 * 0.393701, 5 * 0.393701), 
                             permutation_inv = False):

        def update_plot(i, data, plist, blist):

            for j in range(len(plist)):
                plist[j].center = data[i,j]

                if j < len(data[i]) - 1:
                    P1 = data[i][j]
                    P2 = data[i][j+1]
                    blist[j].set_data([P1[0], P2[0]], [P1[1], P2[1]])
                    
            return plist, blist

        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i in range(len(color_list)):
            color_list[i] = colorConverter.to_rgba(color_list[i], alpha=0.9)

        fig, ax = plt.subplots(1, figsize = fig_size, dpi = 300)

        radius = (np.power(2,1/6) * self.sigma)/2
        particle_list = []
        bond_list = []
        
        
        if smoothing:
            trajectory = trajectory.reshape(trajectory.shape[0], self.n_particles, self.dimensions)
            smoothed_traj = np.zeros([trajectory.shape[0]-smoothing+1, trajectory.shape[1], trajectory.shape[2]])

            for i in range(len(trajectory[0])):
                for j in range(len(trajectory[0,0])):
                    cumsum = np.cumsum(np.insert(trajectory[:,i,j], 0, 0)) 
                    smoothed_traj[:,i,j] = (cumsum[smoothing:] - cumsum[:-smoothing]) / float(smoothing)

            edited_traj = smoothed_traj
        else:
            edited_traj = trajectory = trajectory.reshape(trajectory.shape[0], self.n_particles, self.dimensions)
            
            
        for i, p in enumerate(edited_traj[0]):
            
            color = color_list[i%10]
            
            if permutation_inv:
                color = (0.7,0.7,0.7)
                
            circle = plt.Circle(p, radius, facecolor=color, edgecolor=colorConverter.to_rgba((0,0,0), alpha=0.9), lw=1)
            a = ax.add_patch(circle)
            particle_list.append(a)
            
            if i < len(edited_traj[0]) - 1:
                P1 = edited_traj[0][i]
                P2 = edited_traj[0][i+1] 
                l, = ax.plot([P1[0], P2[0]], [P1[1], P2[1]], zorder=0, c="0", marker="o", markersize=2)
                bond_list.append(l)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        traj_ani = animation.FuncAnimation(fig, update_plot, frames=range(0, len(edited_traj)),
                                  fargs=(edited_traj, particle_list, bond_list), interval=interval, repeat=False)
        plt.close()
        
        return traj_ani
    
    
    def additional_parameters(self):
        
        return {"Epsilon" : self.eps, "Sigma" : self.sigma, "Bond Length" : self.d_bond, "Force Constant Bond" : self.k_bond, 
                "Angle Reference" : self.theta_angle, "Force Constant Angle" : self.k_angle, "Alpha Training" : self.alpha_train}
    