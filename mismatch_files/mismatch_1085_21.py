import torch

input_tensor = torch.randint(1, param1, (param2,), dtype=torch.int64)  # Creating a tensor of non-negative integers
p_value = torch.tensor(param3)  # Probability value
generator = None  # Can be customized if needed

cpu_output = input_tensor.geometric_(p_value.item(), generator=generator)  # on CPU
gpu_output = input_tensor.cuda().geometric_(p_value.item(), generator=generator)  # on GPU

num_of_parameters = 3

# Parameters:
#   - param1: upper limit for the integer values in input tensor : Range = [1, 10000] : Type = int
#   - param2: shape of the input tensor : Range = [1, 10000] : Type = int
#   - param3: probability of success in [0, 1] : Range = [0.0, 1.0] : Type = floatparam1 = int(6000)
param2 = int(333)
param3 = float(0.20)
