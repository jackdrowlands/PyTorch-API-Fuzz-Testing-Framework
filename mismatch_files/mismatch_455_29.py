import torch

# Create a square matrix tensor on CPU
x = torch.randn(param1, param1, dtype=param2)
y = x.cuda()

cpu_output = torch.matrix_exp(x)  # on CPU
gpu_output = torch.matrix_exp(y)  # on GPU

num_of_parameters = 2

# Parameters:
#   - param1: size of the input square matrix (must be a square matrix) : Range = [1, 100] : Type = int
#   - param2: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int8', 'torch.uint8', 'torch.int16', 'torch.int32', 'torch.int64', 'torch.float128', 'torch.complex64', 'torch.complex128'] : Type = torch.dtypeparam1 = int(92)
param2 = torch.float32
