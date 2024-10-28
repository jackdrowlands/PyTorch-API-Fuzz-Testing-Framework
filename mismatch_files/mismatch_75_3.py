import torch
import torch.nn.functional as F

logits = torch.randn(param1, param2, param3, dtype=param4)
y = logits.cuda()

cpu_output = F.gumbel_softmax(logits, tau=param5, hard=param6, eps=param7, dim=param8)  # on CPU
gpu_output = F.gumbel_softmax(y, tau=param5, hard=param6, eps=param7, dim=param8)  # on GPU

num_of_parameters = 8

# Parameters:
#   - param1: first dimension of the logits tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the logits tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the logits tensor : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the logits tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: temperature parameter (tau) for Gumbel softmax : Range = (0, math.inf) : Type = float
#   - param6: boolean flag indicating whether to return the hard version of Gumbel softmax : Range = [True, False] : Type = bool
#   - param7: small positive value to prevent division by zero (epsilon) : Range = (0, math.inf) : Type = float
#   - param8: dimension along which softmax will be computed : Range = [-param1, param1-1] : Type = intparam1 = int(100)
param2 = int(50)
param3 = int(25)
param4 = torch.float32
param5 = float(0.1)
param6 = bool(True)
param7 = float(1e-3)
param8 = int(-2)
