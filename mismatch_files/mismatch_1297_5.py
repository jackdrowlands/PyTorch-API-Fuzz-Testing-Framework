import torch

# Define shape, device, and dtype placeholders
shape = (param1, param2, param3)  # Shape of the tensor
device_cpu = 'cpu'                # Device for CPU
device_gpu = 'cuda'               # Device for GPU
dtype = param4                     # Data type for the tensor

# Create tensors
cpu_output = torch.testing.make_tensor(shape, device=device_cpu, dtype=dtype, 
                                        low=param5, high=param6, 
                                        requires_grad=param7, 
                                        noncontiguous=param8, 
                                        exclude_zero=param9)  # on CPU

gpu_output = torch.testing.make_tensor(shape, device=device_gpu, dtype=dtype, 
                                        low=param5, high=param6, 
                                        requires_grad=param7, 
                                        noncontiguous=param8, 
                                        exclude_zero=param9)  # on GPU

num_of_parameters = 9

# Parameters:
#   - param1: first dimension of the shape : Range = [1, 100] : Type = int
#   - param2: second dimension of the shape : Range = [1, 100] : Type = int
#   - param3: third dimension of the shape (optional) : Range = [1, 100] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.int32', 'torch.int64'] : Type = torch.dtype
#   - param5: lower bound for uniform distribution of values (optional): Range = (-math.inf, math.inf) : Type = float
#   - param6: upper bound for uniform distribution of values (optional): Range = (-math.inf, math.inf) : Type = float
#   - param7: flag indicating whether the tensor requires gradients : Range = [True, False] : Type = bool
#   - param8: flag for creating a non-contiguous tensor : Range = [True, False] : Type = bool
#   - param9: flag to exclude zero from tensor values : Range = [True, False] : Type = boolparam1 = int(100)
param2 = int(100)
param3 = int(100)
param4 = torch.float32
param5 = float(1.0)
param6 = float(10.0)
param7 = bool(True)
param8 = bool(False)
param9 = bool(True)
