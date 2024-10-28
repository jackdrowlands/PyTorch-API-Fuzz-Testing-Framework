import torch

_input_tensor = torch.randn(param1, param2, param3, dtype=param4)
batch1 = torch.randn(param1, param2, param2, dtype=param4)
batch2 = torch.randn(param1, param2, param2, dtype=param4)

cpu_output = _input_tensor.baddbmm_(batch1, batch2, beta=param5, alpha=param6)  # on CPU
_gpu_tensor = _input_tensor.cuda()
gpu_output = _gpu_tensor.baddbmm_(batch1.cuda(), batch2.cuda(), beta=param5, alpha=param6)  # on GPU

num_of_parameters = 6

# Parameters:
#   - param1: an integer representing the batch size : Range = [1, 100] : Type = int
#   - param2: an integer representing the rows of the matrix for batch1 and batch2 : Range = [1, 100] : Type = int
#   - param3: an integer representing the columns of the matrix for _input_tensor : Must be equal to param2 : Range = [1, 100] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensors : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int32'] : Type = torch.dtype
#   - param5: scalar value multiplier for the _input_tensor : Range = (-math.inf, math.inf) : Type = float
#   - param6: scalar value multiplier for the batch matrices : Range = (-math.inf, math.inf) : Type = floatparam1 = int(1)
param2 = int(1)
param3 = int(1)
param4 = torch.float32
param5 = float(0.5)
param6 = float(1.5)
