import torch

tensor = torch.empty(param1, param2, dtype=param3)
tensor_gpu = tensor.clone().cuda()

torch.nn.init.uniform_(tensor, a=param4, b=param5)  # on CPU
torch.nn.init.uniform_(tensor_gpu, a=param4, b=param5)  # on GPU

cpu_output = tensor  # output after uniform initialization on CPU
gpu_output = tensor_gpu  # output after uniform initialization on GPU

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param4: lower bound of the uniform distribution : Range = (-math.inf, math.inf) : Type = float
#   - param5: upper bound of the uniform distribution : Range = (-math.inf, math.inf) : Type = floatparam1 = int(25)
param2 = int(50)
param3 = torch.float64
param4 = float(-1.0)
param5 = float(1.0)
