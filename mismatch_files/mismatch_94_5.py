import torch
import torch.nn.functional as F

input_tensor = torch.randint(0, param1, (param2,), dtype=torch.long)  # Input indices
weight_tensor = torch.randn(param1, param3, dtype=param4)  # Weight matrix
offsets_tensor = torch.tensor([0, param5], dtype=torch.long) if param5 is not None else None  # Optional offsets

cpu_output = F.embedding_bag(input_tensor, weight_tensor, offsets=offsets_tensor, max_norm=param6, 
                              norm_type=param7, scale_grad_by_freq=param8, mode=param9, 
                              sparse=param10, per_sample_weights=None, include_last_offset=param11,
                              padding_idx=param12)  # on CPU

# Move to GPU and calculate
input_tensor_cuda = input_tensor.cuda()
weight_tensor_cuda = weight_tensor.cuda()
offsets_tensor_cuda = offsets_tensor.cuda() if offsets_tensor is not None else None

gpu_output = F.embedding_bag(input_tensor_cuda, weight_tensor_cuda, offsets=offsets_tensor_cuda, 
                              max_norm=param6, norm_type=param7, scale_grad_by_freq=param8, 
                              mode=param9, sparse=param10, per_sample_weights=None, 
                              include_last_offset=param11, padding_idx=param12)  # on GPU

num_of_parameters = 12

# Parameters:
#   - param1: size of the embedding dictionary (vocab size) : Range = [1, 10000] : Type = int
#   - param2: number of indices to lookup : Range = [1, 1000] : Type = int
#   - param3: dimension of the embedding vector : Range = [1, 100] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: optional integer corresponding to the number of offsets (should be less than or equal to param2) : Range = [None, param2] : Type = int or None
#   - param6: optional max norm for embeddings : Range = [0, float('inf')] or None : Type = float or None
#   - param7: norm type for max norm operation : Range = [1, 2] : Type = int
#   - param8: boolean flag to scale gradients by frequency : Range = [True, False] : Type = bool
#   - param9: mode of aggregation : Range = ['mean', 'sum', 'max'] : Type = str
#   - param10: boolean flag for sparse gradient : Range = [True, False] : Type = bool
#   - param11: boolean flag to include last offset : Range = [True, False] : Type = bool
#   - param12: index for padding : Range = [0, param1-1] or None : Type = int or Noneparam1 = int(9999)
param2 = int(800)
param3 = int(89)
param4 = torch.float32
param5 = int(400)
param6 = float(5.0)
param7 = int(2)
param8 = bool(True)
param9 = str('mean')
param10 = bool(False)
param11 = bool(True)
param12 = None
