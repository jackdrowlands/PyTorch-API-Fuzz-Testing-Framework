import torch

input_tensor = torch.empty(param1, param2, param3, dtype=param4)
input_tensor_gpu = input_tensor.cuda()

# In-place operation on CPU
cpu_output = input_tensor.log_normal_(mean=param5, std=param6)  

# In-place operation on GPU
gpu_output = input_tensor_gpu.log_normal_(mean=param5, std=param6)  

num_of_parameters = 6

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: mean of the log-normal distribution : Range = (-math.inf, math.inf) : Type = float
#   - param6: standard deviation of the log-normal distribution : Range = (0, math.inf) : Type = floatparam1 = int(100)
param2 = int(90)
param3 = int(80)
param4 = torch.float32
param5 = float(10.0)
param6 = float(15.0)
