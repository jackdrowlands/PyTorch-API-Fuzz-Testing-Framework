import torch

# Create an input tensor
input_tensor = torch.empty(param1, param2, dtype=param3)
input_tensor_gpu = input_tensor.clone().cuda()  # clone to move to GPU

mean = param4
std = param5
generator = None  # Optional: can be set to a torch.Generator instance

cpu_output = input_tensor.normal_(mean=mean, std=std, generator=generator)  # on CPU
gpu_output = input_tensor_gpu.normal_(mean=mean, std=std, generator=generator)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int32'] : Type = torch.dtype
#   - param4: mean for the normal distribution : Range = (-math.inf, math.inf) : Type = float
#   - param5: standard deviation for the normal distribution : Range = (0, math.inf) : Type = floatparam1 = int(1000)
param2 = int(1000)
param3 = torch.float32
param4 = float(0.5)
param5 = float(4.0)
