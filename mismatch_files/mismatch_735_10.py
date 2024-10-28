import torch

# Creating a random tensor between 0 and 1
x = torch.rand(param1, param2, dtype=param3)
y = x.cuda()

cpu_output = x.bernoulli()  # on CPU
gpu_output = y.bernoulli()  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16'] : Type = torch.dtypeparam1 = int(100)
param2 = int(10000)
param3 = torch.float32
