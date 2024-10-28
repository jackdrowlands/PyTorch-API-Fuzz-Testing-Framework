import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, param3, dtype=param4)
y = x.cuda()

cpu_output = F.rrelu_(x, lower=param5, upper=param6, training=param7)  # on CPU
gpu_output = F.rrelu_(y, lower=param5, upper=param6, training=param7)  # on GPU

num_of_parameters = 7

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor (optional) : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int8', 'torch.uint8', 'torch.int16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param5: lower bound for the random slope : Range = [0, math.inf] : Type = float
#   - param6: upper bound for the random slope : Range = [lower, math.inf] : Type = float
#   - param7: boolean flag indicating whether the module is in training mode : Range = [True, False] : Type = boolparam1 = int(40)
param2 = int(60)
param3 = int(80)
param4 = torch.float32
param5 = float(0.1)
param6 = float(2.2)
param7 = bool(True)
