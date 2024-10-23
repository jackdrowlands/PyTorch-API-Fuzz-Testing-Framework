import torch
import json
param1 = 10
param2 = 0
param3 = -1.0
import torch

# Create an input tensor with random values
input_tensor = torch.randn(param1, param2)  # Placeholder for dimensions
threshold_value = param3  # placeholder for threshold
set_value = param4  # placeholder for value to set

# Move tensor to GPU
input_tensor_gpu = input_tensor.clone().cuda()

# Apply threshold operation on CPU
cpu_output = input_tensor.clone()
cpu_output.threshold_(threshold_value, set_value)  # on CPU

# Apply threshold operation on GPU
gpu_output = input_tensor_gpu.clone()
gpu_output.threshold_(threshold_value, set_value)  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: an integer representing the first dimension of the input tensor
#   - param2: an integer representing the second dimension of the input tensor
#   - param3: a float representing the threshold value
#   - param4: a float representing the value to set for elements below the threshold
print(json.dumps({'cpu_output': cpu_output.tolist(), 'gpu_output': gpu_output.cpu().tolist()}))