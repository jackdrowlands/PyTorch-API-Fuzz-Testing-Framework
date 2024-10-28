import torch

A = torch.randn(param1, param1, dtype=param2)  # Ensure A is a square matrix
n = param3  # the power to raise the matrix
y = A.cuda()

cpu_output = torch.linalg.matrix_power(A, n, out=None)  # on CPU
gpu_output = torch.linalg.matrix_power(y, n, out=None)  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: the number of rows/columns of the square input matrix A : Range = [1, 100] : Type = int
#   - param2: dtype parameter, specifies the data type of the input matrix : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int32', 'torch.int64', 'torch.complex64', 'torch.complex128'] : Type = torch.dtype
#   - param3: an integer or a negative integer representing the power to raise the matrix : Range = [0, 100] or [-100, -1] : Type = intparam1 = int(95)
param2 = torch.float32
param3 = int(50)
