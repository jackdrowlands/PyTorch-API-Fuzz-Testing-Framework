import torch

x = torch.randn(param1, param2, dtype=param3)
y = x.cuda()

cpu_output = x.argsort(dim=param4, descending=param5)  # on CPU
gpu_output = y.argsort(dim=param4, descending=param5)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input tensor : Range = [0, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [0, 10000] : Type = int (if 1D, this can be omitted)
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param4: an integer representing the dimension to sort along : Range = [-param1, param1-1] : Type = int
#   - param5: boolean flag indicating whether to sort in descending order : Range = [True, False] : Type = boolparam1 = int(1234)
param2 = int(4321)
param3 = torch.float32
param4 = int(0)
param5 = bool(False)
