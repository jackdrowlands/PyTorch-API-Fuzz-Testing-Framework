import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(param1, param2)
        self.fc2 = nn.Linear(param2, param3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
amount_to_prune = param4

# Apply L1 Unstructured pruning
prune.l1_unstructured(model.fc1, name='weight', amount=amount_to_prune)  # Prune fc1 weights
prune.l1_unstructured(model.fc2, name='weight', amount=amount_to_prune)  # Prune fc2 weights

# Check the pruned weights
cpu_output = model.fc1.weight.data
gpu_output = model.fc2.weight.data

num_of_parameters = 4

# Parameters:
#   - param1: number of input features for fc1 : Range = [1, 10000] : Type = int
#   - param2: number of output features for fc1 (input features for fc2) : Range = [1, 10000] : Type = int
#   - param3: number of output features for fc2 : Range = [1, 10000] : Type = int
#   - param4: amount to prune (a float representing the proportion or absolute number of parameters to prune) : Range = [0.0, 1.0] or [1, min(param1, param2, param3)] : Type = floatparam1 = int(450)
param2 = int(450)
param3 = int(450)
param4 = float(0.15)
