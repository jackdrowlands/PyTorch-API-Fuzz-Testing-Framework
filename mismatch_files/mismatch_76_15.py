import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, dtype=param3)  # Creating input tensor on CPU
y = x.cuda()  # Moving input tensor to GPU

cpu_output = F.log_softmax(x, dim=param4, dtype=param5)  # on CPU
gpu_output = F.log_softmax(y, dim=param4, dtype=param5)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param4: integer specifying the dimension to compute the softmax (possible values are usually from -len(input.shape) to len(input.shape)-1) : Range = [-len(x.shape), len(x.shape)-1] : Type = int or None
#   - param5: optional parameter specifying the data type for the output tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', None] : Type = torch.dtype or Noneparam1 = int(999)
param2 = int(999)
param3 = torch.float32
param4 = int(-1)
param5 = torch.float64
