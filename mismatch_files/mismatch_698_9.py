import torch

x = torch.randn(param1, param2, dtype=param3)  # Input tensor on CPU
other = torch.tensor(param4, dtype=param3)  # Can be a scalar or tensor

cpu_output = x.gt_(other)  # on CPU

y = x.cuda()  # Move tensor to GPU
other_gpu = other.cuda()  # Move 'other' to GPU if it's a tensor

gpu_output = y.gt_(other_gpu)  # on GPU

num_of_parameters = 4

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype of the input tensor and 'other' : Range = ['torch.float32', 'torch.float64', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param4: 'other' value which can be a scalar or a tensor : Range = [any numeric value] : Type = float/int (scalar) or tensorparam1 = int(1)
param2 = int(9999)
param3 = torch.float32
param4 = float(3.14)
