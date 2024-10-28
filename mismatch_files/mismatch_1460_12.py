import torch

total_count = param1
probs = torch.tensor(param2)  # Example shape (n,) where n is the number of categories
logits = None  # This can be set to a tensor if needed, otherwise None
validate_args = param3

# Create multinomial distribution on CPU
multinomial_cpu = torch.distributions.multinomial.Multinomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args)
cpu_output = multinomial_cpu.sample()

# Move parameters to GPU
probs_gpu = probs.cuda()
multinomial_gpu = torch.distributions.multinomial.Multinomial(total_count=total_count, probs=probs_gpu, logits=logits, validate_args=validate_args)
gpu_output = multinomial_gpu.sample()

num_of_parameters = 3

# Parameters:
#   - param1: an integer representing the total count : Range = [0, 10000] : Type = int
#   - param2: tensor containing probabilities for each category : Range = [0, 1] (must sum to 1) : Type = torch.Tensor (1D tensor)
#   - param3: boolean flag for argument validation : Range = [True, False] : Type = boolparam1 = int(8888)
param2 = torch.tensor([0.15, 0.85], dtype=torch.float32)
param3 = bool(False)
