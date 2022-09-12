import numpy as np
import torch
import yaml

from numba import jit

def write_yaml(file_path, obj):
    
    with open(file_path, "w") as file:
    
        return yaml.dump(obj, file)
    
    
def load_yaml(file_path):
    
    with open(file_path) as file:
    
        return yaml.safe_load(file)

    
def mask_parser(masks):
    mask_str = ["".join(list(masks[i].cpu().numpy().astype(int).astype(str))) for i in range(len(masks))]
    
    result = []
    for s in mask_str:
        for i in range(len(s)):
            substring = s[:i+1]
            if len(s) % len(substring) == 0:
                reconstructed = substring * (len(s) // len(substring))
                if reconstructed == s:
                    result.append(substring + " x {}".format(len(s) // len(substring)))
                    break
            
    return result


def resample(x, log_weights):
    
    assert isinstance(x, (list, torch.Tensor, np.ndarray)), "Reampling argument must be Tensor or list of Tensors."
    
    if isinstance(log_weights, torch.Tensor):
        w = torch.exp(log_weights - torch.max(log_weights)).cpu().numpy()
    elif isinstance(log_weights, np.ndarray):
        w = np.exp(log_weights - np.max(log_weights))
        
    if isinstance(x, list):
        
        N = x[0].shape[0]
        indx = np.random.choice(np.arange(0, N), replace=True, size = N , p = w/np.sum(w))
        
        x_resampled = [part_x[indx] for part_x in x]
        
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        
        N = x.shape[0]

        indx = np.random.choice(np.arange(0, N), replace=True, size = N , p = w/np.sum(w))

        x_resampled = x[indx]
        
    return x_resampled
    

def grouped_mean(values, conditions):
    
    unique_conditions, unique_counts = conditions.unique(return_counts=True)
    
    if len(unique_conditions) == 1:
        return values
    elif len(unique_conditions) == values.shape[0]:
        return values
    
    unique_conditions = unique_conditions.reshape(-1, 1)
    unique_counts = unique_counts.reshape(-1, 1)

    unique_conditions_list = unique_conditions.flatten().tolist()
    conditions_list = conditions.flatten().tolist()

    condition_map = {k:v for k,v in zip(unique_conditions_list, range(len(unique_conditions_list)))}

    indices_np = np.fromiter(map(condition_map.get, conditions_list), dtype=int)
    indices = torch.from_numpy(indices_np).to(values.device).reshape(-1, 1).expand(-1, values.shape[1])

    grouped_mean = torch.zeros(unique_conditions.shape[0], values.shape[1], dtype=torch.float, device=values.device).scatter_add_(0, indices, values) / unique_counts

    return grouped_mean


@jit(nopython=True)
def R_from_vec(u, v):
    
    rot_matrix = np.zeros((2,2))
    
    rot_matrix[0,0] = u[0]*v[0] + u[1]*v[1]
    rot_matrix[0,1] = -(u[0]*v[1] - u[1]*v[0])
    
    rot_matrix[1,0] = u[0]*v[1] - u[1]*v[0]
    rot_matrix[1,1] = u[0]*v[0] + u[1]*v[1]
    
    return rot_matrix


@jit(nopython=True)
def R_from_angle(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
