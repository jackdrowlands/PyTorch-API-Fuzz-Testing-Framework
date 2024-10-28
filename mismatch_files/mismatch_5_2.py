import torch

A = torch.randn(param1, param2, dtype=param3)  # Input matrix
tau = torch.randn(param4, dtype=param5)         # Householder vector

A_cuda = A.cuda()
tau_cuda = tau.cuda()

cpu_output = torch.linalg.householder_product(A, tau)  # on CPU
gpu_output = torch.linalg.householder_product(A_cuda, tau_cuda)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: number of rows in matrix A : Range = [1, 1000] : Type = int
#   - param2: number of columns in matrix A : Range = [1, 1000] : Type = int
#   - param3: dtype parameter, specifies the data type of matrix A : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.complex64', 'torch.complex128'] : Type = torch.dtype
#   - param4: length of the Householder vector tau : Range = [min(param1, param2), min(param1, param2)] : Type = int
#   - param5: dtype parameter for tau, specifies the data type : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.complex64', 'torch.complex128'] : Type = torch.dtypeparam1 = int(1000)
param2 = int(1000)
param3 = torch.float64
param4 = int(1000)
param5 = torch.float64
