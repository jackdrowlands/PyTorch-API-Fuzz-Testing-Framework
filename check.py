import os

# PyTorch API Fuzz Testing Framework - System Check Utility
# This simple utility script checks the number of CPU cores available on the system.
# Used to help determine appropriate thread count for parallel test execution.

num_cores = os.cpu_count()  # Get the number of cores
print(f"Number of CPU cores: {num_cores}")