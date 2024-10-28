import torch

input_tensor = torch.randn(param1, param2, dtype=param3)  # Input tensor
other = param4  # Can be a scalar or tensor

cpu_output = input_tensor.less_equal_(other)  # on CPU
gpu_output = input_tensor.cuda().less_equal_(other)  # on GPU

num_of_parameters = 4

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param4: tensor or scalar to compare against : Range = Any scalar/tensor that matches or broadcasts with input_tensor's shape : Type = scalar or torch.Tensorparam1 = int(300)
param2 = int(750)
param3 = torch.float64
param4 = int(0)
