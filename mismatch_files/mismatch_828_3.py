import torch

# Create a sample tensor with values between 0 and 1
input_tensor_cpu = torch.rand((param1, param2), dtype=param3)
input_tensor_gpu = input_tensor_cpu.clone().cuda()

# Use the bernoulli_ operation on both CPU and GPU
cpu_output = input_tensor_cpu.bernoulli_(p=param4, generator=param5)  # on CPU
gpu_output = input_tensor_gpu.bernoulli_(p=param4, generator=param5)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int32'] : Type = torch.dtype
#   - param4: probability of getting a 1 in the Bernoulli distribution : Range = [0.0, 1.0] : Type = float
#   - param5: optional generator for reproducibility : Range = None or torch.Generator : Type = torch.Generator or Noneparam1 = int(999)
param2 = int(999)
param3 = torch.float32
param4 = float(0.99)
param5 = None
