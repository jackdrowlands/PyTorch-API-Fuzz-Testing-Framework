import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, param3, param4, dtype=param5)
y = x.cuda()

cpu_output = F.dropout2d(x, p=param6, training=param7, inplace=param8)  # on CPU
gpu_output = F.dropout2d(y, p=param6, training=param7, inplace=param8)  # on GPU

num_of_parameters = 8

# Parameters:
#   - param1: batch size (number of images) : Range = [1, 100] : Type = int
#   - param2: number of channels : Range = [1, 64] : Type = int
#   - param3: height of the image : Range = [1, 256] : Type = int
#   - param4: width of the image : Range = [1, 256] : Type = int
#   - param5: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param6: probability of an element to be zeroed : Range = [0.0, 1.0] : Type = float
#   - param7: boolean flag indicating whether to use dropout mode : Range = [True, False] : Type = bool
#   - param8: boolean flag indicating whether to perform the operation in-place : Range = [True, False] : Type = boolparam1 = int(10)
param2 = int(8)
param3 = int(256)
param4 = int(100)
param5 = torch.float64
param6 = float(0.9)
param7 = bool(True)
param8 = bool(False)
