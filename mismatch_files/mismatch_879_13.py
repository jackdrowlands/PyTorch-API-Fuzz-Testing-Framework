import torch

x = torch.randn(param1, param2, dtype=param3)  # Input tensor
other = torch.randn(param1, param2, dtype=param3)  # Comparison tensor

cpu_output = x.greater_(other)  # on CPU
y = x.cuda()
other_gpu = other.cuda()

gpu_output = y.greater_(other_gpu)  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor and comparison tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int8', 'torch.uint8', 'torch.int16', 'torch.int32', 'torch.int64', 'torch.bool'] : Type = torch.dtypeparam1 = int(100)
param2 = int(100)
param3 = torch.float64
