import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, param3, dtype=param4)
y = x.cuda()

cpu_output = F.rrelu(x, lower=param5, upper=param6, training=param7, inplace=param8)  # on CPU
gpu_output = F.rrelu(y, lower=param5, upper=param6, training=param7, inplace=param8)  # on GPU

num_of_parameters = 8

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: lower bound for the random slope : Range = [0.0, 1.0] : Type = float
#   - param6: upper bound for the random slope : Range = [0.0, 1.0] : Type = float
#   - param7: boolean flag for training mode : Range = [True, False] : Type = bool
#   - param8: boolean flag for in-place operation : Range = [True, False] : Type = boolparam1 = int(25)
param2 = int(15)
param3 = int(5)
param4 = torch.float64
param5 = float(0.01)
param6 = float(0.99)
param7 = bool(True)
param8 = bool(True)
