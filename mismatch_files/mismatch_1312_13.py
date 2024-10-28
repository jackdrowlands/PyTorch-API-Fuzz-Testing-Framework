import torch

# Parameters for the Binomial distribution
total_count = param1  # Total count must be a non-negative integer
probs = param2        # Probability of success must be in the range [0, 1]
logits = param3       # Logits can be None or a tensor of float values
validate_args = param4 # Boolean flag to validate arguments

# Create the Binomial distribution
binomial_dist_cpu = torch.distributions.binomial.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args)

# Sample from the distribution on CPU
cpu_output = binomial_dist_cpu.sample()

# Move to GPU and create another Binomial distribution
if torch.cuda.is_available():
    total_count_gpu = total_count  # Keep the same parameters for comparison
    binomial_dist_gpu = torch.distributions.binomial.Binomial(total_count=total_count_gpu, probs=probs, logits=logits, validate_args=validate_args)
    gpu_output = binomial_dist_gpu.sample()
else:
    gpu_output = None  # Indicate that GPU is not available

num_of_parameters = 4

# Parameters:
#   - param1: total_count for the Binomial distribution (Must be non-negative) : Range = [0, 10000] : Type = int
#   - param2: probability of success (Must be in [0, 1]) : Range = [0.0, 1.0] : Type = float or None
#   - param3: logits (Must be None or a tensor of float values) : Range = None or tensor : Type = torch.Tensor or None
#   - param4: validate_args flag (Whether to validate inputs) : Range = [True, False] : Type = boolparam1 = int(9999)
param2 = float(0.5)
param3 = None
param4 = bool(True)
