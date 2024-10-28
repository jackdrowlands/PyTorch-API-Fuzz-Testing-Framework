import torch

# Input tensor and tau tensor creation on CPU
x = torch.randn(param1, param2, dtype=param3)  # input tensor
tau = torch.randn(param4)  # tau tensor (should be of size min(param1, param2))

x_cuda = x.cuda()
tau_cuda = tau.cuda()

cpu_output = torch.orgqr(x, tau)  # on CPU
gpu_output = torch.orgqr(x_cuda, tau_cuda)  # on GPU

num_of_parameters = 4

# Parameters:
#   - param1: number of rows in the input tensor : Range = [1, 10000] : Type = int
#   - param2: number of columns in the input tensor : Range = [1, param1] : Type = int (must be >= number of rows)
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param4: size of tau tensor : Range = [1, min(param1, param2)] : Type = intparam1 = int(150)
param2 = int(150)
param3 = torch.float32
param4 = int(100)
