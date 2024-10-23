import torch
import json
param1 = 1
param2 = 3
param3 = 32
param4 = 32
param5 = 16
param6 = 1
param7 = 3
param8 = 3
import torch

# Input tensor (N, C_in, H_in, W_in)
input_tensor = torch.randn(param1, param2, param3, param4)
# Weight tensor (C_out, C_in / groups, kH, kW)
weight_tensor = torch.randn(param5, param2 // param6, param7, param8)
# Optional bias tensor (C_out)
bias_tensor = None  # You can assign a tensor if needed.

# Move tensors to GPU
input_tensor_gpu = input_tensor.cuda()
weight_tensor_gpu = weight_tensor.cuda()
bias_tensor_gpu = bias_tensor.cuda() if bias_tensor is not None else None

# Perform conv_transpose2d operation on CPU
cpu_output = torch.nn.functional.conv_transpose2d(
    input_tensor,
    weight_tensor,
    bias=bias_tensor,
    stride=param9,
    padding=param10,
    output_padding=param11,
    groups=param12,
    dilation=param13
)

# Perform the same operation on GPU
gpu_output = torch.nn.functional.conv_transpose2d(
    input_tensor_gpu,
    weight_tensor_gpu,
    bias=bias_tensor_gpu,
    stride=param9,
    padding=param10,
    output_padding=param11,
    groups=param12,
    dilation=param13
)

num_of_parameters = 8

# Parameters:
#   - param1: integer for batch size (N)
#   - param2: integer for number of input channels (C_in)
#   - param3: integer for input height (H_in)
#   - param4: integer for input width (W_in)
#   - param5: integer for number of output channels (C_out)
#   - param6: integer for groups (C_in divided by groups)
#   - param7: integer for kernel height (kH)
#   - param8: integer for kernel width (kW)
#   - param9: integer or tuple for stride (default is 1)
#   - param10: integer or tuple for padding (default is 0)
#   - param11: integer or tuple for output_padding (default is 0)
#   - param12: integer for groups (default is 1)
#   - param13: integer or tuple for dilation (default is 1)
print(json.dumps({'cpu_output': cpu_output.tolist(), 'gpu_output': gpu_output.cpu().tolist()}))