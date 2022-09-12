import torch
import numpy as np
from torch.utils.data import Dataset


class transform_dataset(Dataset):
    
    """Basic dataset for the training of the conditional generator."""
    
    def __init__(self, generator,
                 data_tensor, 
                 data_temperature, 
                 test_zx_temperature,
                 condition_tensor,
                 CV_range = None,
                 test_fraction = 0.01,
                 shuffle_data=True,
                 seed=2147483647
                 ):
        """
        Basic dataset for the training of the conditional generator.

        Parameters
        ----------
        generator : conditional_boltzmann.network.networks.conditional_generator
            The generator module for which the dataset is initialized
            
        data_tensor : Tensor
            Data space points as tensor of shape [N, D], where N
            is the number of samples and D the dimension of the data.
            
        data_temperature : Tensor, float or int
            Temperature in units of kB  at which the samples in the data_tensor where generated.
            If a float or int is provided, it is assumed that all samples where generated 
            under constant temperature. If a tensor is provided, a temperature can be specified 
            for each sample. In that case, a tensor of shape [N, 1] must be provided, where N
            is the number of samples.
            
        test_zx_temperature : float or array-like
            Temperature in units of kB for the generation of test temperatures. If a array-like object containing 
            two floats is provided, the testing will be performed at uniforly distributed temperatures 
            between these. Otherwise a constant temperature for tesing is assumed.
            
        condition_tensor : Tensor or None, optional
            Condition under which the samples in the data_tensor where generated. Must be proviced 
            as tensor of shape [B, C], where B is the batch size and C the dimension of the condition.
            May remain default as ``condition_tensor = None`` if the generator is initialized as an
            unconditioned generator.
        
        CV_range [float, float] or None, optional
            Condition min and max for the generation of test conditions assuming a uniform distribution.
            May be ``None`` for unconditioned generation.
                
        test_fraction : float, optional
            Fraction of data_tensor used for testing. The provided value has to be between 0 and 1.
            
        shuffle_data : bool, optional
            Whether data_tensor be shuffled before test data is split of, highly recommended to be True.

        """
        
        #
        # Check Parameters
        #
        
        assert data_tensor.shape[1] == generator.posterior.dofs, \
                "data_tensor should have dimensionality [N, D] where N is \
                the number of samples and D is the number of degrees of freedom."
        
        assert isinstance(data_temperature, (torch.Tensor, float, int)), \
                "Temperature data_temperature must be instance of torch.Tensor, float or int."
        
        if isinstance(data_temperature, (torch.Tensor)):
            assert data_temperature.shape == (data_tensor.shape[0], 1), \
                "data_temperature should have dimensionality [N, 1] where N is the number of samples."
            
        assert test_fraction > 0 and test_fraction < 1, \
                "test_fraction must be greater than 0 and smaller than 1."
        
        assert condition_tensor is not None and CV_range is not None, \
            "condition_tensor and CV_range must be specified when conditioned generation is desired."
        
        assert condition_tensor.shape == (data_tensor.shape[0], generator.condition_dim), \
            "condition_tensor should have dimensionality [N, C] where N is the number of samples and \
            C is condition dimensionality specified in the generator class."
        
        
        N_test = int(test_fraction * data_tensor.shape[0])
        
        generator.eval()
        
        self.prior = generator.prior
        
        random_generator = torch.Generator(device=generator.device)

        if seed is not None:
            random_generator.manual_seed(seed)
        
        #
        # Data
        #
        
        if shuffle_data:

            shuffle_indices = torch.randperm(data_tensor.shape[0], device=generator.device, generator=random_generator)
            
            data_tensor = data_tensor[shuffle_indices]

            if isinstance(data_temperature, (torch.Tensor)):
                data_temperature = data_temperature[shuffle_indices]
            
            condition_tensor = condition_tensor[shuffle_indices]
                
                
        self.train_data_xz = data_tensor[:-N_test]
        self.test_data_xz  = data_tensor[-N_test:]
        
        #
        # Temperature
        #
        
        if isinstance(data_temperature, (torch.Tensor)):
            temperature_tensor = data_temperature
        elif isinstance(data_temperature, (float, int)):
            temperature_tensor = torch.ones((data_tensor.shape[0], 1), dtype=torch.float32, device = generator.device) * data_temperature
            
        self.train_T_xz = temperature_tensor[:-N_test]
        self.test_T_xz  = temperature_tensor[-N_test:]
        
        #
        # Condition
        #

        self.train_c_xz = condition_tensor[:-N_test]
        self.test_c_xz  = condition_tensor[-N_test:]
        
        #
        # z -> x Test Data
        #
        
        if isinstance(test_zx_temperature, (float, int)):
            T_sample = test_zx_temperature
            
        elif isinstance(test_zx_temperature, (tuple, list, np.ndarray)) and len(test_zx_temperature) == 2:
            
            assert generator.prior.multi_T_supported, \
                "For z -> testing at different T the prior must support generation of samples at multiple temperatures."
            
            T_sample = torch.rand([N_test, 1], dtype=torch.float32, device=generator.device, generator=random_generator) 
            T_sample *= (test_zx_temperature[1]-test_zx_temperature[0]) 
            T_sample += test_zx_temperature[0]
        
        else:
            raise Exception("test_zx_temperature must be a float or a list of two floats.")
            
        uniform = torch.rand([N_test, 1], dtype=torch.float32, device=generator.device, generator=random_generator)
        self.test_c_zx = uniform * (CV_range[1]-CV_range[0]) + CV_range[0]

        self.test_data_zx = generator.prior.sample(N_test, T=T_sample, generator=random_generator)
        self.test_T_zx = T_sample
        

    def __len__(self):
        return self.train_data_xz.shape[0]
    
    
    def __getitem__(self, idx):
        
        return self.train_data_xz[idx], self.train_T_xz[idx], self.train_c_xz[idx]
       
    def get_test_data(self):
        
        test_zx = self.test_data_zx
        return self.test_data_xz, self.test_T_xz, self.test_c_xz, test_zx, self.test_T_zx, self.test_c_zx
        
    
    def get_single_temp_test_data(self, reference_temperature):
        
        test_data_xz, test_T_xz, test_c_xz, _, _, test_c_zx = self.get_test_data()

        T_indices = torch.where(test_T_xz == reference_temperature)[0]
        
        N_samples = len(T_indices)
        
        assert N_samples > 0, "No data for given temperature found!"
        
        #
        # Gather Data
        #
        
        T_sample = test_T_xz[T_indices]
        
        x_sample = test_data_xz[T_indices]
        z_sample = self.prior.sample(N_samples, T_sample)
        
        c_sample_xz = test_c_xz[T_indices]
        c_sample_zx = test_c_zx[T_indices]

        #
        # Sum up
        #
        
        energy_test_data = (x_sample, T_sample, c_sample_xz)
        energy_test_data += (z_sample, T_sample, c_sample_zx)

        print("{} test data points for given temperature found.".format(N_samples))
        
        return energy_test_data
