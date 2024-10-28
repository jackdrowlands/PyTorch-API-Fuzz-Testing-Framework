import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, dtype=param3)
y = x.cuda()

cpu_output = F.softplus(x, beta=param4, threshold=param5)  # on CPU
gpu_output = F.softplus(y, beta=param4, threshold=param5)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param4: beta parameter to control the slope of the softplus function : Range = (0, math.inf) : Type = float
#   - param5: threshold parameter to clip very large input values : Range = [0, math.inf] : Type = floatparam1 = int(5120)
param2 = int(213)
param3 = torch.float32
param4 = float(100.0)
param5 = float(300.0)
