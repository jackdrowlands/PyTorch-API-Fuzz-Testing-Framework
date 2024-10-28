import torch



torch.random.manual_seed(param1)  # Setting seed for CPU
cpu_output = torch.rand(3, 3)  # Generate random tensor on CPU

torch.cuda.random.manual_seed(param1)  # Setting seed for GPU
gpu_output = torch.rand(3, 3, device='cuda')  # Generate random tensor on GPU

num_of_parameters = 1

# Parameters:
#   - param1: integer value for the seed : Range = [0, 2**32-1] : Type = intparam1 = int(54321)
