import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, param3, dtype=param4)
y = x.cuda()

cpu_output = F.alpha_dropout(x, p=param5, training=param6, inplace=param7)  # on CPU
gpu_output = F.alpha_dropout(y, p=param5, training=param6, inplace=param7)  # on GPU

num_of_parameters = 7

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: float parameter representing the probability of an element to be zeroed out : Range = [0.0, 1.0] : Type = float
#   - param6: boolean flag indicating whether the model is in training mode : Range = [True, False] : Type = bool
#   - param7: boolean flag indicating whether to perform the operation in-place : Range = [True, False] : Type = boolparam1 = int(100)
param2 = int(100)
param3 = int(100)
param4 = torch.float64
param5 = float(0.8)
param6 = bool(True)
param7 = bool(False)
