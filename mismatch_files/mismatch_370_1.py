import torch

x = torch.randn(param1, param2, dtype=param3)
y = x.cuda()

cpu_output, cpu_indices = torch.topk(x, k=param4, dim=param5, largest=param6, sorted=param7)  # on CPU
gpu_output, gpu_indices = torch.topk(y, k=param4, dim=param5, largest=param6, sorted=param7)  # on GPU

num_of_parameters = 7

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int8', 'torch.uint8', 'torch.int16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param4: the number of top elements to retrieve : Range = [1, min(param1, param2)] : Type = int
#   - param5: the dimension along which to find the top k : Range = [0, 1] (or None for default) : Type = int or None
#   - param6: boolean flag to select the largest or smallest values : Range = [True, False] : Type = bool
#   - param7: boolean flag to indicate if the results should be sorted : Range = [True, False] : Type = boolparam1 = int(128)
param2 = int(256)
param3 = torch.float32
param4 = int(128)
param5 = int(0)
param6 = bool(True)
param7 = bool(False)
