import torch

x = torch.randn(param1, param2, param3, dtype=param4)
y = x.cuda()

cpu_output = torch.argsort(x, dim=param5, descending=param6)  # on CPU
gpu_output = torch.argsort(y, dim=param5, descending=param6)  # on GPU

num_of_parameters = 6

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param5: integer representing the dimension to sort along : Range = [-3, 2] (for a 3D tensor) : Type = int
#   - param6: boolean flag indicating whether to sort in descending order : Range = [True, False] : Type = boolparam1 = int(10000)
param2 = int(50)
param3 = int(30)
param4 = torch.float32
param5 = int(-3)
param6 = bool(False)
