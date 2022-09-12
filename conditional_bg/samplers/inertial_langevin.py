import numpy as np

from numba import jit


class inertial_langevin_integrator(object):
    
    def __init__(self, model, length, dt, gamma, m, T):
        
        self.model = model
        self.dofs = model.dofs
        
        self.length = length
        self.dt = dt
        self.gamma = gamma
        self.m = m
        self.T = T
    
    @staticmethod
    @jit(nopython=True) 
    def step(prev_x, prev_v, dt, F, T, m, gamma):

        g = np.random.randn(prev_x.shape[0]).astype(np.float32)

        alpha = np.float32(1.0-np.exp(-gamma*dt))
        
        v_prime = prev_v + (F * dt)/m
        
        dv = -alpha * v_prime + np.sqrt( (T/m) * alpha * (2-alpha) ) * g

        x = prev_x + ( v_prime + (dv/2) ) * dt
        v = v_prime + dv
        
        return x, v
    
    @staticmethod
    @jit(nopython=True) 
    def integrate_n_steps(x, v, dt, T, m, gamma, 
                          path_length, stride, force_function, 
                          integrator, model_parameters, remove_com_motion):
        
        path = np.zeros((path_length//stride+1, x.shape[0]), dtype=np.float32)
        
        velocities = np.zeros((path_length//stride+1, x.shape[0]), dtype=np.float32)
        last_x = x
        last_v = v
        
        path[0] = last_x
        velocities[0] = last_v
        
        for i in range(1, path_length):
            
            F = force_function(last_x, model_parameters)
            
            x, v = integrator(last_x, last_v, dt, F, T, m, gamma)
            
            if remove_com_motion:
                x[::2] -= x[::2].sum()/(x.shape[0]/2)
                x[1::2] -= x[1::2].sum()/(x.shape[0]/2)
            
            
            if i%stride == 0:
                path[i//stride] = x
                velocities[i//stride] = v
                
            
            last_x = x.astype(np.float32)
            last_v = v.astype(np.float32)
                
        return path, velocities
    
        
    @staticmethod
    @jit(nopython=True) 
    def integrate_to_state(x, v, dt, T, m, gamma, 
                          max_length,stride, force_function, 
                          integrator, model_parameters, 
                          state_function, remove_com_motion):
        
        path = np.zeros((max_length//stride+1, x.shape[0]), dtype=np.float32)
        
        velocities = np.zeros((max_length//stride+1, x.shape[0]), dtype=np.float32)
        
        last_x = x
        last_v = v
        
        path[0] = last_x
        velocities[0] = last_v
        
        for i in range(1, max_length):
            
            F = force_function(last_x, model_parameters)
            
            x, v = integrator(last_x, last_v, dt, F, T, m, gamma)
            
            if remove_com_motion:
                x[::2] -= x[::2].sum()/(x.shape[0]/2)
                x[1::2] -= x[1::2].sum()/(x.shape[0]/2)
            
            
            if i%stride == 0:
                path[i//stride] = x
                velocities[i//stride] = v

                if state_function(x.astype(np.float32)) != 10:
                    return path[:i//stride+1], velocities[:i//stride+1]
            
            last_x = x.astype(np.float32)
            last_v = v.astype(np.float32)
            
            
    
        return path[:max_length//stride], velocities[:max_length//stride]
    
    def generate_flexible_path(self, init_x, init_v, state_function, max_length, stride, remove_com_motion=False):

        init_x = np.array(init_x, dtype=np.float32)
        
        init_v = np.array(init_v, dtype=np.float32)

            
        model_parameters = self.model.get_model_parameters()
        
        path, velocities = self.integrate_to_state(init_x, init_v, 
                                                  self.dt, self.T, 
                                                  self.m, self.gamma, 
                                                  max_length, stride, self.model._force, 
                                                  self.step, model_parameters,
                                                  state_function, remove_com_motion)
        
        return path, velocities
    
    
    
    def generate_path(self, init_x = None, init_v = None, stride = 1, remove_com_motion=False):
        
        if np.any(init_x == None):
            init_x = np.zeros(self.dofs, dtype=np.float32)
        else:
            init_x = np.array(init_x, dtype=np.float32)
           
        if np.any(init_v == None):
            init_v = np.zeros(self.dofs, dtype=np.float32)
        else:
            init_v = np.array(init_v, dtype=np.float32)
        
        model_parameters = self.model.get_model_parameters()
        
        path, velocities = self.integrate_n_steps(init_x, init_v, 
                                                  self.dt, self.T, 
                                                  self.m, self.gamma, 
                                                  self.length, stride, self.model._force, 
                                                  self.step, model_parameters,
                                                  remove_com_motion)
        
        
        return path, velocities