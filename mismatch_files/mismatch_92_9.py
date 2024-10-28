import torch
import torch.nn.functional as F

# Creating a 3D tensor of shape (batch_size, channels, depth, height, width)
x = torch.randn(param1, param2, param3, param4, dtype=param5)
y = x.cuda()

cpu_output = F.dropout3d(x, p=param6, training=param7, inplace=param8)  # on CPU
gpu_output = F.dropout3d(y, p=param6, training=param7, inplace=param8)  # on GPU

num_of_parameters = 8

# Parameters:
#   - param1: batch size of the input tensor : Range = [1, 100] : Type = int
#   - param2: number of channels : Range = [1, 100] : Type = int
#   - param3: depth of the input tensor : Range = [1, 100] : Type = int
#   - param4: height of the input tensor : Range = [1, 100] : Type = int
#   - param5: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param6: probability of an element to be zeroed (0<=p<=1) : Range = [0.0, 1.0] : Type = float
#   - param7: boolean flag indicating whether the module is in training mode : Range = [True, False] : Type = bool
#   - param8: boolean flag indicating whether to perform the operation in-place : Range = [True, False] : Type = boolparam1 = int(99)
param2 = int(37)
param3 = int(34)
param4 = int(6)
param5 = torch.float32
param6 = float(0.9)
param7 = bool(True)
param8 = bool(True)
