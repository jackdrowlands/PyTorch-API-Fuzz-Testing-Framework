import torch

low = param1
high = param2
size = (param3, param4)  # size should be a tuple
generator = None         # Optional
out = None               # Optional, for output tensor
dtype = param5           # Optional
layout = torch.strided   # Optional, default layout
device = 'cpu'          # Move to GPU later
requires_grad = False    # Optional

cpu_output = torch.randint(low=low, high=high, size=size, generator=generator, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)  # on CPU

device = 'cuda'         # Move to GPU for GPU output
gpu_output = torch.randint(low=low, high=high, size=size, generator=generator, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: low value (inclusive) for the random integers : Range = [0, high) : Type = int
#   - param2: high value (exclusive) for the random integers : Range = [1, 1000] : Type = int
#   - param3: first dimension of the output tensor size : Range = [1, 100] : Type = int
#   - param4: second dimension of the output tensor size : Range = [1, 100] : Type = int
#   - param5: dtype parameter, specifies the data type of the output tensor : Range = ['torch.int32', 'torch.int64', 'torch.uint8'] : Type = torch.dtypeparam1 = int(10)
param2 = int(950)
param3 = int(100)
param4 = int(5)
param5 = torch.float32
