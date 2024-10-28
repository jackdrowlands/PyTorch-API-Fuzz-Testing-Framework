import torch

input_tensor = torch.randn(param1, param2, param3, dtype=param4)  # Creating input tensor
input_tensor_gpu = input_tensor.cuda()  # Moving tensor to GPU

cpu_output = torch.randint_like(input_tensor, low=param5, high=param6, dtype=param7)  # on CPU
gpu_output = torch.randint_like(input_tensor_gpu, low=param5, high=param6, dtype=param7)  # on GPU

num_of_parameters = 7

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param4: dtype parameter for the input tensor : Range = ['torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param5: lower bound for the random integers : Range = [0, param6) : Type = int
#   - param6: upper bound for the random integers : Range = (param5, 10000): Type = int
#   - param7: optional dtype for the output tensor : Range = ['torch.int32', 'torch.int64'] : Type = torch.dtypeparam1 = int(1500)
param2 = int(250)
param3 = int(75)
param4 = torch.float64
param5 = int(500)
param6 = int(800)
param7 = torch.int64
