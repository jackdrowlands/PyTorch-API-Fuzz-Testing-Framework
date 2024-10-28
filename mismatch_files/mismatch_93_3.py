import torch
import torch.nn.functional as F

input_indices = torch.randint(0, param1, (param2,), dtype=torch.long)  # Input indices for embedding
weight = torch.randn(param1, param3, dtype=param4)  # Weight matrix for embeddings

input_indices_gpu = input_indices.cuda()
weight_gpu = weight.cuda()

cpu_output = F.embedding(input_indices, weight, padding_idx=param5, max_norm=param6, norm_type=param7, scale_grad_by_freq=param8, sparse=param9)  # on CPU
gpu_output = F.embedding(input_indices_gpu, weight_gpu, padding_idx=param5, max_norm=param6, norm_type=param7, scale_grad_by_freq=param8, sparse=param9)  # on GPU

num_of_parameters = 9

# Parameters:
#   - param1: size of the embedding (number of embeddings) : Range = [1, 10000] : Type = int
#   - param2: number of indices to retrieve : Range = [1, 10000] : Type = int
#   - param3: size of each embedding vector : Range = [1, 1024] : Type = int
#   - param4: dtype parameter, specifies the data type of the embeddings : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: index for padding to ignore during the embedding : Range = [0, param1] : Type = int (optional)
#   - param6: maximum norm for embeddings : Range = [0.0, math.inf] : Type = float (optional)
#   - param7: type of norm for max_norm : Range = [1.0, 2.0] : Type = float (optional)
#   - param8: whether to scale gradients by frequency : Range = [True, False] : Type = bool (optional)
#   - param9: whether to use sparse gradients : Range = [True, False] : Type = bool (optional)param1 = int(7500)
param2 = int(500)
param3 = int(1024)
param4 = torch.float32
param5 = int(2500)
param6 = float(1.0)
param7 = float(1.5)
param8 = bool(True)
param9 = bool(False)
