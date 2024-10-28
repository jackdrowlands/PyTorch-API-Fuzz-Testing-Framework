import torch

input_tensor = torch.randn(param1, param2, dtype=param3)
other = torch.randn(param1, param2, dtype=param3).to(input_tensor.device)  # ensure same device

cpu_output = input_tensor.le_(other)  # on CPU
input_tensor_gpu = input_tensor.clone().cuda()
gpu_output = input_tensor_gpu.le_(other.cuda())  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int8', 'torch.uint8', 'torch.int16', 'torch.int32', 'torch.int64'] : Type = torch.dtypeparam1 = int(300)
param2 = int(700)
param3 = torch.float32
