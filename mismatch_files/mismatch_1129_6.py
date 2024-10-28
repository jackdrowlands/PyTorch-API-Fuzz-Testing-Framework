import torch

x = torch.randn(param1, param2, dtype=param3)  # Ensure x is a 2D tensor
tau = torch.randn(param4, dtype=param3)  # Tau should be a 1D tensor matching x's shape criteria

y = x.cuda()
tau_gpu = tau.cuda()

cpu_output = torch.orgqr(x, tau)  # on CPU
gpu_output = torch.orgqr(y, tau_gpu)  # on GPU

num_of_parameters = 4

# Parameters:
#   - param1: first dimension of the input tensor (number of rows) : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor (number of columns, must be >= param1) : Range = [param1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param4: the size of tau, should be equal to param1 : Range = [param1] : Type = intparam1 = int(1000)
param2 = int(1000)
param3 = torch.float64
param4 = int(1000)
