import torch

# Create input tensor with values between 0 and 1
x = torch.rand(param1, param2, dtype=param3)
y = x.cuda()

cpu_output = torch.bernoulli(x, generator=param4, out=param5)  # on CPU
gpu_output = torch.bernoulli(y, generator=param4, out=param5)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter for the input probabilities : Range = ['torch.float32', 'torch.float64'] : Type = torch.dtype
#   - param4: generator for random number generation (can be None) : Range = [None, torch.Generator] : Type = Union[None, torch.Generator]
#   - param5: optional output tensor where the result will be stored (can be None) : Range = [None, torch.Tensor] : Type = Union[None, torch.Tensor]param1 = int(50)
param2 = int(150)
param3 = torch.float32
param4 = None
param5 = None
