import torch

param1 = 100
param2 = torch.float32
param3 = bool(True)

# Generating a square matrix A on CPU
A = torch.randn(param1, param1, dtype=param2)
A = A @ A.T  # Make it positive definite to avoid singular issues

# Moving tensor A to GPU
A_cuda = A.cuda()

# Invoking torch.linalg.inv_ex on CPU
cpu_output = torch.linalg.inv(A, out=None)

# Invoking torch.linalg.inv_ex on GPU
gpu_output = torch.linalg.inv(A_cuda, out=None)

num_of_parameters = 3

# Parameters:
#   - param1: an integer representing the dimension of the square matrix : Range = [1, 100] : Type = int
#   - param2: dtype parameter, specifies the data type of matrix A : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param3: boolean flag indicating whether to check for errors during inversion : Range = [True, False] : Type = boolparam1 = int(15)

print(torch.allclose(cpu_output, gpu_output.cpu(), rtol=1e-01, atol=1e-03, equal_nan=False))
print(A)
print(cpu_output)
print(gpu_output.cpu())
# print(torch.round(cpu_output - gpu_output.cpu(), decimals=1).tolist())