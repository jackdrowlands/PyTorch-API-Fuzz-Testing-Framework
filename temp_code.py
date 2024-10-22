import torch
import json
param1 = 3
param2 = 3
param3 = 3
param4 = 1
param5 = 5
param6 = 10
param7 = None
import torch
import numpy as np










m = torch.nn.Conv2d(param1, param2, param3)
input = torch.randn(param4, param5, param6, param7)
cpu_output = torch.tanh(m(input))
cpu_output = torch.max(cpu_output)

m.cuda()
input = input.cuda()
gpu_output = torch.tanh(m(input))
gpu_output = torch.max(gpu_output)


num_of_parameters = 7
print(json.dumps({'cpu_output': cpu_output.tolist(), 'gpu_output': gpu_output.cpu().tolist()}))