import numpy as np

import torch

from conditional_bg import util
from conditional_bg.network.coupling_layers import affine_coupling_layer
    
class base_generator(torch.nn.Module):

    def __init__(self, dim, prior, posterior, coupling_blocks, device, coupling_layer_params):
        
        super(base_generator, self).__init__()
       
        #
        # Initialize general parameters
        #
        self.device = device
        
        self.dimensions = dim
        
        self.prior = prior
        self.posterior = posterior
        
        
        #
        # Initialize bias and condition.
        #
        self._biased = False
        self.condition_function = None
        self.likelihood = None
        
        self._conditioned = False
        self.condition_dim = None
        self.norm_range_condition = None
        
        #
        # Initialize transform layers
        #
        self._transform = False
        self.transform_layers = []
        self._angle_loss = False
        

        #
        # Initialize flow layers
        #
        self.coupling_block_descriptor = coupling_blocks
        self.n_coupling_blocks = len(coupling_blocks)
        self.n_blocks = None # defined later in self.initialize_flow
        
        self.coupling_layer_params = coupling_layer_params
        self.realNVP_blocks = []
            
        
    def initialize_flow(self):
        
        coupling_layer_dict = {"A" : affine_coupling_layer}

        block_list = []

        for B in self.coupling_block_descriptor:
            
            coupling_layer = coupling_layer_dict[B] 


            network_unit = coupling_layer(dimension = self.dimensions,
                                          condition_dim = self.condition_dim,
                                          **self.coupling_layer_params[B])

            block_list.append(network_unit.to(self.device))
            
        self.n_blocks = len(block_list)
        self.realNVP_blocks = torch.nn.ModuleList(block_list)
        
            
    def initialize_transform(self):
        
        # If this function is called, transformation_layers is not empty so we set _transform to True
        self._transform = True
        
        transform_angle_losses = [t.in_torsion_loss for t in self.transform_layers]
        
        # If any layer has an angle loss term, set _angle_loss to True
        if any(transform_angle_losses):
            self._angle_loss = True

        dim_reduction = [t.delta_dofs for t in self.transform_layers]

        # Add/subtract dropped/added degrees of freedom from dimensionality of the generator
        self.dimensions += np.sum(dim_reduction)
        
    
    def normalize_condition(self, condition):

        # Normalize condition vector
        condition_NN = 2 * ((condition - self.norm_range_condition[0]) / (self.norm_range_condition[1]-self.norm_range_condition[0])) - 1

        return condition_NN
    
    
    def F(self, direction, x, T=None, condition=None):
        
        raise NotImplementedError
            
    
    def loss(self, direction, x, T, condition=None, clamp_max = None, rc_loss_data=None):
        
        raise NotImplementedError
    
    
    def get_angle_loss(self, x_untransformed, condition=None, transform_layers=None):
        
        torsion_loss = x_untransformed.new_zeros(1)
        
        transform_layers = transform_layers if transform_layers is not None else self.transform_layers
        
        # Iterate through transform layers
        for i in reversed(range(len(transform_layers))):
            
            if transform_layers[i].in_torsion_loss:
                
                # Get angles and find values outside [-pi, pi]
                angles = transform_layers[i].get_angles(x_untransformed)
                
                high_torsion_loss = torch.where(angles > np.pi,  (angles-np.pi)**2, torch.zeros((1,1), dtype=torch.float32, device=self.device))
                low_torsion_loss  = torch.where(angles < -np.pi, (angles+np.pi)**2, torch.zeros((1,1), dtype=torch.float32, device=self.device))

                torsion_loss = (high_torsion_loss + low_torsion_loss).sum(dim=-1, keepdims=True)
            
            # If we are not at the last transformation, we apply the current transformation and go on
            # Remember we go reversed, so i=0 is the last transformation
            if i > 0:
                
                if transform_layers[i].conditioned:
                    x_untransformed, _ = transform_layers[i].F_zx(x_untransformed, condition, return_Jacobian=False)
                else:
                    x_untransformed, _ = transform_layers[i].F_zx(x_untransformed, return_Jacobian=False)
                    
        return torsion_loss.mean()
    
    
    def get_invalid_indices(self, x_untransformed, condition=None, transform_layers=None):
        
        transform_layers = transform_layers if transform_layers is not None else self.transform_layers
        
        invalid_sample_indices = []
        
        if self._angle_loss and x_untransformed is not None:
            
            for i in reversed(range(len(transform_layers))):
            
                if transform_layers[i].in_torsion_loss:
                
                    angles = transform_layers[i].get_angles(x_untransformed)
                    
                    break

                if i > 0:
                    if transform_layers[i].conditioned:
                        x_untransformed, _ = transform_layers[i].F_zx(x_untransformed, condition)
                    else:
                        x_untransformed, _ = transform_layers[i].F_zx(x_untransformed)
                        
            invalid_samples =  (angles > np.pi) | (angles < -np.pi)
            
            invalid_sample_indices = torch.where(torch.any(invalid_samples, dim=-1) == 1)[0]
        
        return invalid_sample_indices
    
    
    def sample(self, N, T, batch_size = None, ref_condition=None, MCMC_steps=None, MCMC_displacement=0.1, reference_latent=None, resample=False, as_numpy=False):
        
        raise NotImplementedError
            
    
    def log_path_weight(self, path, ref_condition, T, generated_distribution=False):
        
        if not generated_distribution:
            
            generated_condition = self.condition_function(path.clone())

            logL =  -(1/T) * self.likelihood.log_prob(generated_condition, ref_condition)

            log_weight =  -torch.logsumexp(logL, dim=0)
        
        else:
            
            # Transform path
            T_tensor = torch.ones(path.shape[0], 1, device=self.device, dtype=torch.float32) * T
            cond_tensor = torch.ones(path.shape[0], 1, device=self.device, dtype=torch.float32) * ref_condition
            
            path_z, _, _, log_det_zx = self.F("xz", path, T=T_tensor, condition=cond_tensor)
            
            p_gen =  -(1/T) * self.prior.log_prob(path_z) + log_det_zx
            
            # eq_prb
            eq_prb = -(1/T) * self.posterior.log_prob(path)
            
            log_weight = -torch.logsumexp(p_gen-eq_prb, dim=0)
            
        return log_weight
    
    
    def clamp(self, a, min_val=None, max_val=None, clamp_val=1e20):
       
        if max_val is None and min_val is None:
            raise Exception("At least one of min_val or max_val value should be defined.")
        
        if max_val is not None:
            
            a_clamped = torch.where(((a < max_val) | (a > clamp_val)), 
                            a, 
                            max_val + torch.log(a - max_val + 1) )

            a_clamped = torch.where( a_clamped < clamp_val,
                            a_clamped, 
                            torch.tensor(max_val + np.log(clamp_val - max_val + 1), dtype=torch.float32, device=self.device))
            
            
        if min_val is not None:
            
            a_clamped = torch.where(((a_clamped > min_val) | (a_clamped < -clamp_val)), 
                            a_clamped, 
                            min_val - torch.log(min_val - a_clamped + 1))

            a_clamped = torch.where(a_clamped > -clamp_val, 
                            a_clamped, 
                            torch.tensor(min_val - np.log(min_val + clamp_val + 1), dtype=torch.float32, device=self.device))

        
        a_clamped[torch.isnan(a)] = torch.tensor(max_val + np.log(clamp_val - max_val + 1), dtype=torch.float32, device=self.device)
        
        return a_clamped

    
    def __str__(self):
        
        msg  = "--------------------------------------------------------------\n"
        msg += "                     Generator Summary                        \n"
        msg += "--------------------------------------------------------------\n\n"
        
        msg += "Parameters: {}\n".format(sum(p.numel() for p in self.parameters() if p.requires_grad))
        msg += "Device: {}\n".format(self.device)
        msg += "DOFs: {}\n".format(self.dimensions)
        
        
        msg += "\n--------------------------------------------------------------\n"
        msg += "Prior: {}\n".format(self.prior.__class__.__name__)
        msg += "--------------------------------------------------------------\n"
        
        prior_properties = self.prior.parameter_dict()
        for prop in prior_properties:
            msg += "\t {}: {}\n".format(prop, prior_properties[prop])
            
        msg += "\n--------------------------------------------------------------\n"
        msg += "Posterior: {}\n".format(self.posterior.__class__.__name__)
        msg += "--------------------------------------------------------------\n"
        posterior_properties = self.posterior.parameter_dict()
        for prop in posterior_properties:
            msg += "\t {}: {}\n".format(prop, posterior_properties[prop])
        
        
        msg += "\n--------------------------------------------------------------\n"
        msg += "Network Architecture:\n"
        msg += "--------------------------------------------------------------\n"
        
        msg += "\t Coupling Layers: {}\n".format(self.coupling_block_descriptor)
        
        properties = self.realNVP_blocks[0].parameter_dict()
        for prop in properties:
            
            if prop == "Masks":
                m_str = properties[prop]
                msg += "\t\t Masks: \t{}\n".format(m_str[0])
                
                for m in m_str[1:]:
                    msg += "\t\t\t\t\t{}\n".format(m)
            else:
                msg += "\t\t {}: {}\n".format(prop, properties[prop])
            
        msg += "\n\t Coupling Network Parameters:\n"
        properties = self.realNVP_blocks[0].t[0].parameter_dict()
        for prop in properties:
            msg += "\t\t {}: {}\n".format(prop, properties[prop])
            
        msg += "\n\t Coupling Blocks: {}\n".format(self.n_coupling_blocks)
        msg += "\t Total Blocks: {}\n".format(self.n_blocks)
        
        
        #self.coupling_layer_params
        
        msg += "\n--------------------------------------------------------------\n"
        msg += "Network Properties:\n"
        msg += "--------------------------------------------------------------\n"
        
        msg += "\t Biased: {}\n".format(self._biased)
        msg += "\t Conditioned: {}\n".format(self._conditioned)
        
        if self._biased:
            msg += "\n\t Likelihood Function: {}\n".format(self.likelihood.__class__.__name__)
            properties = self.likelihood.parameter_dict()
            for prop in properties:
                msg += "\t\t {}: {}\n".format(prop, properties[prop])
                
            msg += "\n\t Condition Function: {}\n".format(self.condition_function.__name__)
            
        if self._conditioned:
            msg += "\t Condition Dimension: {}\n".format(self.condition_dim)
            msg += "\t Condition Range: {} - {}\n".format(self.norm_range_condition[0], self.norm_range_condition[1])
            
        
        msg += "\t Angle Loss: {}\n".format(self._angle_loss)
        
        
        msg += "\n--------------------------------------------------------------\n"
        msg += "Transformations: {}\n".format(self._transform)
        msg += "--------------------------------------------------------------\n"
        if self._transform:
            for i, t in enumerate(self.transform_layers):
                msg += "\t Layer {}: {}\n".format(i, t.__class__.__name__)
        
                
        return msg