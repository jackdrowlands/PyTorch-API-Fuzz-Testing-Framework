import torch

x = torch.abs(torch.randn(param1, param2, dtype=param3))  # ensuring non-negative input for poisson
if param4 is not None:
    generator = torch.manual_seed(param4)
else:
    generator = None

cpu_output = torch.poisson(x, generator=generator)  # on CPU
gpu_output = torch.poisson(x.cuda(), generator=generator)  # on GPU

num_of_parameters = 4

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16'] : Type = torch.dtype
#   - param4: optional integer seed for random number generation : Range = [0, 2**32-1] : Type = int (None allowed)param1 = int(700)
param2 = int(800)
param3 = torch.float32
param4 = None
