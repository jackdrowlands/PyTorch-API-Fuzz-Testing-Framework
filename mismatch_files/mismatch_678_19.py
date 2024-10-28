import torch

# Create a square matrix A
A = torch.randn(param1, param1, dtype=param2)

# Create an input tensor with compatible dimensions
input_tensor = torch.randn(param1, param3, dtype=param2)

# Move both to GPU
A_gpu = A.cuda()
input_tensor_gpu = input_tensor.cuda()

cpu_output = torch.linalg.solve(A, input_tensor)  # on CPU
gpu_output = torch.linalg.solve(A_gpu, input_tensor_gpu)  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: dimension of the square matrix A : Range = [1, 10000] : Type = int
#   - param2: dtype parameter, specifies the data type of the input tensor and matrix A : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param3: second dimension of the input tensor (Must be compatible with A) : Range = [1, 10000] : Type = intparam1 = int(9000)
param2 = torch.float32
param3 = int(88)
