import torch

# Create a tensor with random values
x = torch.randn(param1, param2, param3)
y = x.clone().cuda()  # Clone and move to GPU

# Apply exponential_ operation on CPU
cpu_output = x.exponential_(lambd=param4)

# Apply exponential_ operation on GPU
gpu_output = y.exponential_(lambd=param4)

num_of_parameters = 4

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param4: lambda parameter for exponential distribution : Range = [0, math.inf] : Type = floatparam1 = int(125)
param2 = int(125)
param3 = int(125)
param4 = float(0.001)
