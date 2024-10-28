import torch

# Ensure the tensor is square for matrix_exp
input_tensor = torch.randn(param1, param1, dtype=param2)
input_tensor_gpu = input_tensor.cuda()

cpu_output = input_tensor.matrix_exp()  # on CPU
gpu_output = input_tensor_gpu.matrix_exp()  # on GPU

num_of_parameters = 2

# Parameters:
#   - param1: an integer representing the dimension of the square input tensor : Range = [1, 1000] : Type = int
#   - param2: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.complex64', 'torch.complex128'] : Type = torch.dtypeparam1 = int(999)
param2 = torch.float32
