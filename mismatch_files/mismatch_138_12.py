import torch

cpu_output = torch.rand(param1, param2, param3, dtype=param4, device='cpu', requires_grad=param5)  # on CPU
gpu_output = torch.rand(param1, param2, param3, dtype=param4, device='cuda', requires_grad=param5)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: size of the first dimension of the output tensor : Range = [1, 10000] : Type = int
#   - param2: size of the second dimension of the output tensor : Range = [1, 10000] : Type = int
#   - param3: size of the third dimension of the output tensor (optional) : Range = [1, 10000] : Type = int
#   - param4: specifies the data type of the output tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: boolean flag indicating whether the tensor requires gradient : Range = [True, False] : Type = boolparam1 = int(9000)
param2 = int(20)
param3 = int(40)
param4 = torch.float32
param5 = bool(True)
