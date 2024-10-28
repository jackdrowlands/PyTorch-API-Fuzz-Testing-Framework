import torch

mean = torch.tensor(param1, dtype=param2)  # mean of the normal distribution
std = torch.tensor(param3, dtype=param2)    # standard deviation of the normal distribution
size = (param4, param5)                      # size of the output tensor

cpu_output = torch.normal(mean, std, size)  # on CPU
gpu_output = torch.normal(mean.cuda(), std.cuda(), size)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: mean of the normal distribution : Range = (-math.inf, math.inf) : Type = float
#   - param2: dtype parameter, specifies the data type for mean and std : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param3: standard deviation of the normal distribution (must be non-negative) : Range = [0, math.inf) : Type = float
#   - param4: size dimension 1 of the output tensor : Range = [1, 10000] : Type = int
#   - param5: size dimension 2 of the output tensor : Range = [1, 10000] : Type = intparam1 = float(-1.23)
param2 = torch.float64
param3 = float(0.5)
param4 = int(10)
param5 = int(10)
