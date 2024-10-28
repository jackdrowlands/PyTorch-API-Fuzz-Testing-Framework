import torch

# Input tensor should be of shape (num_classes) with non-negative values.
input_tensor = torch.randn(param1).abs()  # using abs() to avoid negative weights
input_tensor = input_tensor / input_tensor.sum()  # normalizing to sum to 1

num_samples = param2
replacement = param3
generator = None  # You can create a torch.Generator() object for reproducibility if needed.
out = None  # Placeholder for potential output tensor

cpu_output = torch.multinomial(input_tensor, num_samples, replacement, generator=generator, out=out)  # on CPU
gpu_tensor = input_tensor.cuda()
gpu_output = torch.multinomial(gpu_tensor, num_samples, replacement, generator=generator, out=out)  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: number of classes for the multinomial distribution : Range = [2, 10000] : Type = int (must be >= 2)
#   - param2: number of samples to draw : Range = [1, param1] : Type = int
#   - param3: boolean flag indicating whether sampling is done with replacement : Range = [True, False] : Type = boolparam1 = int(2)
param2 = int(2)
param3 = bool(True)
