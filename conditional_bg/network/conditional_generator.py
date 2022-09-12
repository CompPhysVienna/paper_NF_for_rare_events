import numpy as np

import os
import warnings

import torch

from conditional_bg import util
from conditional_bg.network.networks import base_generator

    
class conditional_generator(base_generator):

    def __init__(self, dim, prior, posterior, coupling_blocks, device, coupling_layer_params, 
                 condition_function, condition_dim, likelihood, norm_range_condition, transform_layers=[]):
        
        super(conditional_generator, self).__init__(dim, prior, posterior, coupling_blocks, device, coupling_layer_params)
        
        #
        # Initialize bias and condition.
        #
        
        self._biased = True
        self.condition_function = condition_function
        self.likelihood = likelihood
        
        self._conditioned = True
        self.condition_dim = condition_dim
        self.norm_range_condition = norm_range_condition
        
        #
        # Initialize transform layers
        #
        self.transform_layers = torch.nn.ModuleList(transform_layers)
        
        if len(self.transform_layers) > 0:
            self.initialize_transform()
        
        #
        # Initialize flow layers
        #
        
        self.initialize_flow()
        
    
    def F(self, direction, x, condition):
        
        log_det_J = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=self.device)
        
        # Normalize condition vector
        condition_NN = self.normalize_condition(condition)
        
        z_untransformed = x.clone()
            
        # Apply transform layers without gradient tracking
        if self._transform and direction == "xz":
            
            with torch.no_grad():
                
                for i in range(len(self.transform_layers)):

                    if self.transform_layers[i].conditioned:
                        z_untransformed, part_det_J = self.transform_layers[i].F_xz (z_untransformed, condition)
                    else:
                        z_untransformed, part_det_J = self.transform_layers[i].F_xz (z_untransformed)

                    log_det_J += part_det_J
        
        
        # Apply realNVP blocks to sample
        block_order = reversed(range(self.n_blocks)) if direction == "xz" else range(self.n_blocks)
        
        for i in block_order:
            
            if direction == "xz":
                transform_function = self.realNVP_blocks[i].F_xz
            else:
                transform_function = self.realNVP_blocks[i].F_zx
                
            z_untransformed, part_det_J = transform_function(z_untransformed, condition_NN)
                
            log_det_J += part_det_J

        
        z_transformed = z_untransformed.clone()
        
        
        # Apply transform layers in reverse order
        if self._transform and direction == "zx":
            
            for i in reversed(range(len(self.transform_layers))):
                
                # Allow for conditioned transformations which take the condition vector as argument
                if self.transform_layers[i].conditioned:
                    z_transformed, part_det_J = self.transform_layers[i].F_zx(z_transformed, condition)
                else:
                    z_transformed, part_det_J = self.transform_layers[i].F_zx(z_transformed)
                
                log_det_J += part_det_J
                
            
        return z_transformed, z_untransformed, log_det_J
    
    
    def loss(self, direction, x, T, condition, clamp_max = None):
        
        assert direction in ["xz", "zx"], "Loss can only be calculated for direction \"xz\" or \"zx\"."
        
        # Transform sample, if user-defined transformations are present the untransformed sample is retained
        z, z_untransformed, log_det_J = self.F(direction, x, condition)
        
        if direction == "xz":
            log_prob_z = self.prior.log_prob(z) 
        else:
            log_prob_z = self.posterior.log_prob(z)
        
        # calculate the bias contribution 
        log_L = 0
        if direction == "zx":
            generated_condition = self.condition_function(z)
            log_L = self.likelihood.log_prob(generated_condition, condition)
       
        log_prob = (1 / T) * (log_prob_z + log_L)

        # Clamp energies
        if clamp_max and direction == "zx": 
            log_prob = self.clamp(log_prob, min_val=None, max_val=clamp_max)
            
        loss = log_prob - log_det_J
        
        loss_angle = None
        
        if direction == "zx":
            
            # Get angle loss if it is included
            if self._angle_loss:
                loss_angle = self.loss_angle(z_untransformed, condition)
        
        return loss, loss_angle, None, None, (log_prob * T)
    
        
    def loss_angle(self, x_untransformed, condition):
        
        loss_angle = self.get_angle_loss(x_untransformed, condition=condition)
        
        return loss_angle.mean()
    
    
    def log_w(self, x, z, log_det_Jzx, T, ref_condition, x_untransformed = None):
        
        # Get bias contribution
        generated_condition = self.condition_function(x.clone())
        
        log_L = self.likelihood.log_prob(generated_condition, ref_condition)
            
        log_prob_x = self.posterior.log_prob(x)
        
        log_prob_z = self.prior.log_prob(z)
            
        # Log weight calculation
        log_w = -(1/T) * (log_prob_x + log_L) 
        log_w += (1/T) * log_prob_z
        log_w += log_det_Jzx
        
        # Set weights for configurations with internal angles outside [-pi, pi] to -inf
        
        invalid_sample_indices = self.get_invalid_indices(x_untransformed, ref_condition)
        log_w[invalid_sample_indices] = -np.inf
            
        return log_w
    
    
    def sample(self, N, T, batch_size = None, ref_condition=None, reference_latent=None, resample=False, as_numpy=False):
        
        self.eval()
        
        if reference_latent is not None:
            assert N == reference_latent.shape[0], "Specified N and reference latent variable size do not match."
            assert batch_size is None, "Batching not supported with reference latent variable."
        
        if batch_size is None:
            batch_size = N
            
        if isinstance(T, (float, int)):
            T = torch.ones((N, 1), dtype=torch.float32, device=self.device) * T
            
        n_batches = int(np.ceil(N/batch_size))
        
        z, x, log_w = [], [], []
        
        with torch.no_grad():
            for i in range(n_batches):
                
                # Initialize condition tensor
                condition_tensor = ref_condition[i*batch_size : (i+1) * batch_size]
               
                T_sample = T[i*batch_size : (i+1) * batch_size]

                if reference_latent is None:
                    
                    z_part = self.prior.sample(batch_size, T = T_sample)
                    
                else:
                    
                    z_part = reference_latent[i*batch_size : (i+1) * batch_size]
                

                # Transform points
                x_part, x_untransformed, log_det_Jzx = self.F("zx", z_part, condition=condition_tensor)

                # Get weights
                log_w_part = self.log_w(x_part, z_part, log_det_Jzx, T=T_sample, ref_condition=condition_tensor, 
                                        x_untransformed=x_untransformed)
                    
                if as_numpy:
                    z_part = z_part.cpu().numpy()
                    x_part = x_part.cpu().numpy()
                    log_w_part = log_w_part.cpu().numpy()
                    
                z.append(z_part)
                x.append(x_part)
                log_w.append(log_w_part)
            
            if as_numpy:
                z = np.vstack(z)
                x = np.vstack(x)
                log_w = np.vstack(log_w)
                    
            else:
                z = torch.vstack(z)
                x = torch.vstack(x)
                log_w = torch.vstack(log_w)
                
            
        return_values = [z, x, log_w]
            
        if resample:
            z_re, x_re, log_w_re  = util.resample([z, x, log_w], log_w.reshape(-1))
            return_values += [z_re, x_re, log_w_re]
            
        return return_values
        
    
    def umbrella_sampling(self, N_samples, T, umbrella_centers, bins, resample=False, batch_size=None, estimator_args={}, outlier_handling="drop"):

        umbrella_centers = list(umbrella_centers)
        force_constants = [self.likelihood.k for _ in range(len(umbrella_centers))]

        cv_timeseries = []
        bin_timeseries = []
        
        
        if isinstance(T, (float, int)):
            T_sample = torch.ones((N_samples, 1), dtype=torch.float32, device=self.device) * T
        elif isinstance(T, torch.Tensor):
            T_sample = T
            
            
        for i, center in enumerate(umbrella_centers):

            with torch.no_grad():

                condition = torch.ones((N_samples, 1), dtype=torch.float32, device=self.device) * center
                
                if resample:
                    
                    *_, z, x, log_w = self.sample(N_samples, T_sample, ref_condition=condition, 
                                                  batch_size=batch_size, resample=True)
                else:
                    z, x, log_w = self.sample(N_samples, T_sample, ref_condition=condition, 
                                              batch_size=batch_size, resample=False)
                    
                generated_condition = self.condition_function(x).cpu().numpy()

            cv_timeseries.append(generated_condition.astype(np.float64))

            bin_ts = np.digitize(generated_condition, bins=bins).astype(int)
            
            if np.any(bin_ts == 0) or np.any(bin_ts == len(bins)):
                warnings.warn("Values below/above specified bin range detected, they will be clipped or dropped")
                
            if outlier_handling == "clip":
                bin_ts = np.clip(bin_ts, 1, len(bins)-1) - 1
            elif outlier_handling == "drop":
                cv_timeseries[-1] = cv_timeseries[-1][(bin_ts > 0) & (bin_ts < len(bins))]
                bin_ts = bin_ts[(bin_ts > 0) & (bin_ts < len(bins))] - 1
                
            
            bin_timeseries.append(bin_ts.ravel())

            
        import pyemma
        
        wham = pyemma.thermo.estimate_umbrella_sampling(cv_timeseries, bin_timeseries, umbrella_centers, force_constants, kT=T, **estimator_args)
        
        bin_centers = bins[:-1] + (bins[1] - bins[0])/2
        
        f = np.full(len(bin_centers), np.inf)
        f[wham.active_set] = wham.f
        
        return bin_centers, f