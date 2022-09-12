import numpy as np

def __make_hash__(d):

    check = ''
        
    for k in d:
        check += str(d[k])
        
    return hash(check)


class biased_metropolis_monte_carlo(object):
    
    def __init__(self, model):
        
        self.model = model
        self.dofs = model.dofs
        
        self.x_last = None
        
        self._cache = {}


    def __cache__(self, n_samples, T, n_cycles, step, cv, likelihood, ref, shuffle):

        self._cache = {
            "n_samples" : n_samples,
            "T" : T,
            "n_cycles" : n_cycles,
            "step" : step,
            "cv" : cv,
            "likelihood" : likelihood,
            "ref" : ref,
            "shuffle" : shuffle
        }
    
        return

        
    def __check_cache__(self, n_samples, T, n_cycles, step, cv, likelihood, ref, shuffle):
        
        if self._cache:
            
            checksum = __make_hash__(self._cache)
            
            d = {
                "n_samples" : n_samples,
                "T" : T,
                "n_cycles" : n_cycles,
                "step" : step,
                "cv" : cv,
                "likelihood" : likelihood,
                "ref" : ref,
                "shuffle" : shuffle
            }
            
            if checksum != __make_hash__(d):
                self.__cache__(n_samples, T, n_cycles, step, cv, likelihood, ref, shuffle)
                return False
        
        else:
        
            self.__cache__(n_samples, T, n_cycles, step, cv, likelihood, ref, shuffle)
        
        return True
    
        
    def metropolis_cycle(self, x, u_x, T, step):
        
        n_samples = np.shape(x)[0]

        shift = np.zeros([n_samples, self.dofs])
        selected_particles = np.random.randint(self.model.n_particles, size=n_samples)
        selected_dofs = np.linspace(selected_particles*self.model.dimensions, (selected_particles+1)*self.model.dimensions, num=self.model.dimensions, endpoint=False, dtype=int)
        shift[np.arange(len(shift)), selected_dofs] = (np.random.random(np.shape(selected_dofs))*2 - 1)*step
        xp = x + shift
        
        u_xp = self.model.potential_energy_batch(xp)
        mask = np.random.random(size=n_samples) < np.exp(-1./T*(u_xp - u_x))
        acc = mask.sum()/len(mask)
        x[mask] = xp[mask]
        u_x[mask] = u_xp[mask]
        
        return x, u_x, acc

    
    def bias_metropolis_cycle(self, x, u_x, T, step, cv, likelihood, ref):
        
        n_samples = np.shape(x)[0]
        
        shift = np.zeros([n_samples, self.dofs])
        selected_particles = np.random.randint(self.model.n_particles, size=n_samples)
        selected_dofs = np.linspace(selected_particles*self.model.dimensions, (selected_particles+1)*self.model.dimensions, num=self.model.dimensions, endpoint=False, dtype=int)
        shift[np.arange(len(shift)), selected_dofs] = (np.random.random(np.shape(selected_dofs))*2 - 1)*step
        xp = x + shift

        u_xp = self.model.potential_energy_batch(xp) + likelihood.bias(cv(xp), ref)
        mask = np.random.random(size=n_samples) < np.exp(-1./T*(u_xp - u_x))
        acc = mask.sum()/len(mask)
        
        x[mask] = xp[mask]
        u_x[mask] = u_xp[mask]
        
        return x, u_x, acc

    
    def sample_space(self, n_samples, T, n_cycles, step, cv, likelihood, ref, shuffle):
        
        same_system = self.__check_cache__(n_samples, T, n_cycles, step, cv, likelihood, ref, shuffle)
        
        if self.x_last is None or not same_system:
            x = np.array([self.model.initial_configuration() for i in range(n_samples)])
        else:
            x = self.x_last

                
        if likelihood is not None:
            
            u_x = self.model.potential_energy_batch(x) + likelihood.bias(cv(x), ref)
                
            for cycle in range(n_cycles):
                x, u_x, acc = self.bias_metropolis_cycle(x, u_x, T, step, cv, likelihood, ref)
        
        else:
            
            u_x = self.model.potential_energy_batch(x)
                
            for cycle in range(n_cycles):
                x, u_x, acc = self.metropolis_cycle(x, u_x, T, step)            
                        
        if shuffle:
            np.random.shuffle(x)
        
        self.x_last = x
                
        return x, u_x, acc