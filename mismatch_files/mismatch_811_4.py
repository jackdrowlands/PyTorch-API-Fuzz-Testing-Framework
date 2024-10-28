import torch

# Placeholder for tensors and parameters
input_tensor = torch.randn(param1, param2, dtype=param3)
index = torch.randint(0, param1 * param2, (param4,), dtype=torch.long)  # Random indices
source = torch.randn(param4, dtype=param3)  # Source tensor to populate based on index
accumulate = param5  # Boolean flag for accumulation

# Perform the put_ operation
cpu_output = input_tensor.clone()  # Cloning to avoid modifying the original tensor
cpu_output.put_(index, source, accumulate)

gpu_input_tensor = input_tensor.cuda()
gpu_source = source.cuda()
gpu_output = gpu_input_tensor.clone()  # Cloning to avoid modifying the original tensor
gpu_output.put_(index.cuda(), gpu_source, accumulate)

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input_tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input_tensor : Range = [1, 10000] : Type = int
#   - param3: dtype parameter (data type of the input and source tensors) : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.int32'] : Type = torch.dtype
#   - param4: number of indices to gather : Range = [1, param1 * param2] : Type = int
#   - param5: boolean flag for whether to accumulate values at the indices : Range = [True, False] : Type = boolparam1 = int(800)
param2 = int(600)
param3 = torch.float64
param4 = int(4800)
param5 = bool(False)
