import torch
import torch.nn as nn

# Sample input tensor with 5 dimensions, e.g., (N, C, D, H, W)
x = torch.randn(param1, param2, param3, param4, param5, dtype=param6)
y = x.cuda()

dropout = nn.Dropout3d(p=param7, inplace=param8)

cpu_output = dropout(x)  # on CPU
gpu_output = dropout(y)  # on GPU

num_of_parameters = 8

# Parameters:
#   - param1: first dimension (N) of the input tensor : Range = [1, 100] : Type = int
#   - param2: second dimension (C) of the input tensor (channels) : Range = [1, 100] : Type = int
#   - param3: third dimension (D) of the input tensor (depth) : Range = [1, 100] : Type = int
#   - param4: fourth dimension (H) of the input tensor (height) : Range = [1, 100] : Type = int
#   - param5: fifth dimension (W) of the input tensor (width) : Range = [1, 100] : Type = int
#   - param6: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param7: float value for the probability of an element to be zeroed (dropout probability) : Range = (0.0, 1.0) : Type = float
#   - param8: boolean flag indicating whether to perform the operation in-place : Range = [True, False] : Type = boolparam1 = int(100)
param2 = int(1)
param3 = int(99)
param4 = int(2)
param5 = int(4)
param6 = torch.float64
param7 = float(0.99)
param8 = bool(True)
