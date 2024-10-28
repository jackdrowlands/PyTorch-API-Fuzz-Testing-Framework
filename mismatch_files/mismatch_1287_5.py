import torch

total_count = param1  # Total number of successes
probs = torch.tensor(param2, dtype=param3) if param2 is not None else None  # Probability of success
logits = torch.tensor(param4, dtype=param3) if param4 is not None else None  # Log-odds of success

# Initialize NegativeBinomial distribution
nb_dist_cpu = torch.distributions.negative_binomial.NegativeBinomial(total_count, probs=probs, logits=logits)

# Sample from the distribution
cpu_output = nb_dist_cpu.sample((param5,))  # Sample size defined by param5
gpu_output = nb_dist_cpu.sample((param5,)).cuda()  # Sample on GPU

num_of_parameters = 5

# Parameters:
#   - param1: total_count, an integer (number of successes) : Range = [1, 10000] : Type = int
#   - param2: tensor-like input for probabilities (optional) : Range = [0.0, 1.0] : Type = float tensor or None
#   - param3: dtype for probs/logits : Range = ['torch.float32', 'torch.float64'] : Type = torch.dtype
#   - param4: tensor-like input for logits (optional) : Range = [-math.inf, +math.inf] : Type = float tensor or None
#   - param5: integer representing the number of samples to draw : Range = [1, 100] : Type = intparam1 = int(600)
param2 = None
param3 = torch.float32
param4 = torch.tensor([-3.5, 1.5], dtype=torch.float32)
param5 = int(1)
