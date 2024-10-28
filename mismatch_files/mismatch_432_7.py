import torch

input_tensor = torch.randn(param1, param2, param3, dtype=param4)
batch1 = torch.randn(param1, param2, param3, dtype=param4)
batch2 = torch.randn(param1, param3, param3, dtype=param4)

y_cpu = torch.baddbmm(input_tensor, batch1, batch2, beta=param5, alpha=param6)  # on CPU
y_gpu = torch.baddbmm(input_tensor.cuda(), batch1.cuda(), batch2.cuda(), beta=param5, alpha=param6)  # on GPU

cpu_output = y_cpu
gpu_output = y_gpu

num_of_parameters = 6

# Parameters:
#   - param1: the batch size of the input tensors : Range = [1, 100] : Type = int
#   - param2: the first dimension of the matrices (must match with the second dimension of batch1 and batch2) : Range = [1, 100] : Type = int
#   - param3: the second dimension of the matrices : Range = [1, 100] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensors : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: scalar value used for beta in the operation : Range = (-math.inf, math.inf) : Type = float
#   - param6: scalar value used for alpha in the operation : Range = (-math.inf, math.inf) : Type = floatparam1 = int(1)
param2 = int(2)
param3 = int(3)
param4 = torch.float32
param5 = float(-math.inf)
param6 = float(math.inf)
