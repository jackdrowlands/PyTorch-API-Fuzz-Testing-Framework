import torch

_input_tensor = torch.empty(param1, param2, dtype=param3)
index = torch.randint(0, param4, (param1, param5), dtype=torch.long)  # Index tensor for scatter
src = torch.rand(param1, param5, dtype=param3)                        # Source tensor for the scatter

cpu_output = _input_tensor.scatter(dim=param6, index=index, src=src)  # on CPU
gpu_output = _input_tensor.cuda().scatter(dim=param6, index=index.cuda(), src=src.cuda())  # on GPU

num_of_parameters = 6

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 100] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 100] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.int64'] : Type = torch.dtype
#   - param4: maximum value for index tensor elements : Range = [1, param1] : Type = int
#   - param5: second dimension for index and src tensors : Range = [1, 10] : Type = int
#   - param6: dimension along which to scatter : Range = [0, 1] : Type = int (must be less than the number of dimensions of _input_tensor)param1 = int(20)
param2 = int(73)
param3 = torch.float64
param4 = int(20)
param5 = int(4)
param6 = int(0)
