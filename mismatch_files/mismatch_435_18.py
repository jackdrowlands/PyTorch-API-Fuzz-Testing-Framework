import torch

# Create a positive definite matrix
def create_positive_definite_matrix(size):
    A = torch.randn(size, size)
    return A @ A.T  # Ensuring it's positive definite

size = param1
x = create_positive_definite_matrix(size)
y = x.cuda()

cpu_output = torch.cholesky(x, upper=param2)  # on CPU
gpu_output = torch.cholesky(y, upper=param2)  # on GPU

num_of_parameters = 2

# Parameters:
#   - param1: size of the square matrix : Range = [1, 100] : Type = int
#   - param2: boolean flag indicating whether to return the upper-triangular factor : Range = [True, False] : Type = boolparam1 = int(90)
param2 = bool(False)
