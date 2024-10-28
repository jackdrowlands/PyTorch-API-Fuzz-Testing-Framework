import torch

# Initialize ShortStorage with different parameters
cpu_output = torch.ShortStorage(param1)  # on CPU with param1
gpu_output = torch.ShortStorage(param2)  # on CPU with param2

num_of_parameters = 2

# Parameters:
#   - param1: size of the storage, must be a non-negative integer : Range = [0, 10000] : Type = int
#   - param2: initialization values, can be a list, tuple or other iterable containing valid short integers : Range = [-32768, 32767] : Type = list/tuple of intparam1 = int(87)
param2 = list([-32768])
