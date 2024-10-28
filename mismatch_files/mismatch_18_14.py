import torch

# Creating a symmetric positive definite matrix A
# For testing, we can create a random matrix and multiply it by its transpose
A = torch.randn(param1, param1, dtype=param2) @ torch.randn(param1, param1, dtype=param2)

# Move tensor to GPU
A_gpu = A.cuda()

# Perform cholesky_ex operation on CPU
cpu_output, cpu_info = torch.linalg.cholesky_ex(A, upper=param3, check_errors=param4)

# Perform cholesky_ex operation on GPU
gpu_output, gpu_info = torch.linalg.cholesky_ex(A_gpu, upper=param3, check_errors=param4)

num_of_parameters = 4

# Parameters:
#   - param1: dimension of the input square tensor A : Range = [1, 100] : Type = int
#   - param2: dtype parameter, specifies the data type of the input tensor A : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param3: boolean flag indicating whether to compute the upper triangular factor : Range = [True, False] : Type = bool
#   - param4: boolean flag indicating whether to check for errors in the input : Range = [True, False] : Type = boolparam1 = int(20)
param2 = torch.float64
param3 = bool(True)
param4 = bool(False)
