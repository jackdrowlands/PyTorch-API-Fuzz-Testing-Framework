import torch

# Define size parameters for the tensor
size_params = (param1, param2, param3)  # Expandable to any number of dimensions

# Create tensor on CPU
cpu_output = torch.randn(*size_params, dtype=param4, device='cpu', requires_grad=param5) 

# Create tensor on GPU
gpu_output = torch.randn(*size_params, dtype=param4, device='cuda', requires_grad=param5)

num_of_parameters = 5

# Parameters:
#   - param1: an integer representing the first dimension of the output tensor : Range = [0, 100] : Type = int
#   - param2: an integer representing the second dimension of the output tensor : Range = [0, 100] : Type = int
#   - param3: an integer representing the third dimension of the output tensor : Range = [0, 100] : Type = int (optional, can be ignored for 1D or 2D tensors)
#   - param4: dtype parameter, specifies the data type of the output tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: boolean flag to specify if gradients should be tracked : Range = [True, False] : Type = boolparam1 = int(10)
param2 = int(20)
param3 = int(30)
param4 = torch.float32
param5 = bool(True)
