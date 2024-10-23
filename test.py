import torch

param1 = 100
param2 = 100
param3 = torch.float64

A = torch.randn(param1, param2, dtype=param3)
tau = torch.randn(param2, dtype=param3)
A_gpu = A.cuda()
tau_gpu = tau.cuda()

cpu_output = torch.linalg.householder_product(A, tau)  # on CPU
gpu_output = torch.linalg.householder_product(A_gpu, tau_gpu)  # on GPU

print(A)
print(tau)
print(cpu_output)
print(gpu_output)