import torch
from torch.distributions import Normal, Independent

# Create a base distribution
base_distribution = Normal(loc=torch.randn(param1), scale=torch.abs(torch.randn(param1)))

# Create an Independent distribution
independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=param2, validate_args=param3)

cpu_output = independent_distribution.rsample()  # Sampling from CPU
gpu_output = independent_distribution.sample().cuda()  # Sampling from GPU

num_of_parameters = 3

# Parameters:
#   - param1: number of samples or dimensions for the base distribution : Range = [1, 10000] : Type = int
#   - param2: an integer representing the number of reinterpreted batch dims : Range = [0, param1] : Type = int
#   - param3: boolean flag for validating arguments : Range = [None, True, False] : Type = Optional[bool]param1 = int(1)
param2 = int(1)
param3 = bool(False)
