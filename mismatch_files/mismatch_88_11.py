import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, param3, dtype=param4)
y = x.cuda()

cpu_output = F.dropout(x, p=param5, training=param6, inplace=param7)  # on CPU
gpu_output = F.dropout(y, p=param5, training=param6, inplace=param7)  # on GPU

num_of_parameters = 7

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int8', 'torch.uint8', 'torch.int16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param5: probability of an element to be zeroed (dropout probability) : Range = [0.0, 1.0] : Type = float
#   - param6: boolean flag indicating whether the module is in training mode : Range = [True, False] : Type = bool
#   - param7: boolean flag indicating whether to perform the operation in-place : Range = [True, False] : Type = boolparam1 = int(250)
param2 = int(450)
param3 = int(80)
param4 = torch.float32
param5 = float(0.5)
param6 = bool(True)
param7 = bool(False)
