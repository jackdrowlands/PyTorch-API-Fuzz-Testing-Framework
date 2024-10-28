import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, dtype=param3)
y = x.clone().cuda()  # clone to avoid in-place modification on GPU

cpu_output = F.rrelu_(x, lower=param4, upper=param5, training=param6)  # on CPU
gpu_output = F.rrelu_(y, lower=param4, upper=param5, training=param6)  # on GPU

num_of_parameters = 6

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param4: a float lower bound for the random slope in positive regions : Range = (0.0, 1.0) : Type = float
#   - param5: a float upper bound for the random slope in positive regions : Range = (0.0, 1.0) : Type = float
#   - param6: boolean flag indicating whether the module is in training mode : Range = [True, False] : Type = boolparam1 = int(2048)
param2 = int(1024)
param3 = torch.float32
param4 = float(0.05)
param5 = float(0.95)
param6 = bool(True)
