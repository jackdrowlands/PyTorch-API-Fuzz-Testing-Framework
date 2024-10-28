import torch

# Create a positive definite matrix for testing
A = torch.rand(param1, param1, dtype=param2)
A = A @ A.T  # Ensure the matrix is positive definite

# Move matrix to GPU
B = A.cuda()

cpu_output = torch.linalg.cholesky(A, upper=param3)  # on CPU
gpu_output = torch.linalg.cholesky(B, upper=param3)  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: dimension of the input square matrix : Range = [1, 100] : Type = int
#   - param2: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param3: boolean flag indicating whether to return the upper triangular matrix : Range = [True, False] : Type = boolparam1 = int(47)
param2 = torch.float32
param3 = bool(True)
