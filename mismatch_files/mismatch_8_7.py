import torch
param1 = 100
param2 = torch.float32
param3 = int(-100)
A = torch.randn(param1, param1, dtype=param2)  # Create a square matrix
n = param3  # Power to raise the matrix to
A_gpu = A.cuda()

cpu_output = torch.linalg.matrix_power(A, n)  # on CPU
gpu_output = torch.linalg.matrix_power(A_gpu, n)  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: size of the square matrix (A) : Range = [1, 100] : Type = int
#   - param2: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param3: integer power to which the matrix is raised : Range = [-100, 100] : Type = intparam1 = int(100)

print(torch.allclose(cpu_output, gpu_output.cpu(), rtol=1e-03, atol=1e-05, equal_nan=False))
print(A)
print(cpu_output)
print(gpu_output.cpu())