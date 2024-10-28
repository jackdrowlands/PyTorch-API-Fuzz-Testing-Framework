import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as param_utils

# Define a sample module, e.g., a Linear layer
module = nn.Linear(param1, param2)  # param1: input features, param2: output features

# Apply orthogonal parameterization on CPU
param_utils.orthogonal(module, name='weight', orthogonal_map=param3, use_trivialization=param4)
cpu_output = module.weight.data.clone()  # Weight after parameterization

# Move the module to GPU
module = module.cuda()

# Apply the same operation on GPU
param_utils.orthogonal(module, name='weight', orthogonal_map=param3, use_trivialization=param4)
gpu_output = module.weight.data.clone()  # Weight after parameterization

num_of_parameters = 4

# Parameters:
#   - param1: number of input features for the module : Range = [1, 10000] : Type = int
#   - param2: number of output features for the module : Range = [1, 10000] : Type = int
#   - param3: an optional custom orthogonal map, or None : Range = [None, any valid callable] : Type = callable
#   - param4: boolean flag indicating whether to use trivialization : Range = [True, False] : Type = boolparam1 = int(1024)
param2 = int(512)
param3 = None
param4 = bool(False)
