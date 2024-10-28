import torch

# Assuming integer types for the bitwise operations
x = torch.randint(0, 256, (param1, param2), dtype=param3)  # Create a random integer tensor
shift_value = param4  # Value to shift by

# Perform left shift operation on CPU and GPU
cpu_output = x.bitwise_left_shift_(shift_value)  # on CPU
y = x.cuda()
gpu_output = y.bitwise_left_shift_(shift_value)  # on GPU

num_of_parameters = 4

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter, specifies the data type of the input tensor : Range = ['torch.int8', 'torch.int16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param4: integer amount to shift left : Range = [0, 31] (for typical int32 representation) : Type = intparam1 = int(64)
param2 = int(64)
param3 = torch.int32
param4 = int(4)
