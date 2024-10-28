import torch
import torch.nn.functional as F

logits = torch.randn(param1, param2, param3, dtype=param4)  # Shape of logits tensor
y = logits.cuda()

cpu_output = F.gumbel_softmax(logits, tau=param5, hard=param6, eps=param7, dim=param8)  # on CPU
gpu_output = F.gumbel_softmax(y, tau=param5, hard=param6, eps=param7, dim=param8)  # on GPU

num_of_parameters = 8

# Parameters:
#   - param1: first dimension of the logits tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the logits tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the logits tensor (optional) : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the logits tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: temperature parameter : Range = (0, math.inf) : Type = float
#   - param6: boolean flag indicating whether to return one-hot encoded output (hard) : Range = [True, False] : Type = bool
#   - param7: small numerical value to prevent division by zero : Range = (0, math.inf) : Type = float
#   - param8: dimension along which to perform the softmax operation : Range = [-len(logits.shape), len(logits.shape)-1] : Type = intparam1 = int(3)
param2 = int(3)
param3 = int(3)
param4 = torch.float64
param5 = float(2.5)
param6 = bool(False)
param7 = float(0.002)
param8 = int(0)
