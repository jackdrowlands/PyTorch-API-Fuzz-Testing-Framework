import torch

A = torch.randn(param1, param2, dtype=param3)  # Input tensor
tau = torch.randn(param4, dtype=param3)         # Householder reflector coefficients
A_gpu = A.cuda()
tau_gpu = tau.cuda()

cpu_output = torch.linalg.householder_product(A, tau)  # on CPU
gpu_output = torch.linalg.householder_product(A_gpu, tau_gpu)  # on GPU

num_of_parameters = 4

# Parameters:
#   - param1: first dimension of the input tensor A : Range = [1, 1000] : Type = int
#   - param2: second dimension of the input tensor A : Range = [1, 1000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.complex64', 'torch.complex128'] : Type = torch.dtype
#   - param4: dimension of the tau tensor, which should match the first dimension of A : Range = [1, min(param1, param2)] : Type = intparam1 = int(700)
param2 = int(200)
param3 = torch.float32
param4 = int(200)
