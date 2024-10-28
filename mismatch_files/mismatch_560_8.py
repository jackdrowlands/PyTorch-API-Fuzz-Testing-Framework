import torch
import torch.nn as nn

# Define RNN parameters
input_size = param1  # Size of input features
hidden_size = param2  # Number of features in the hidden state
num_layers = param3  # Number of recurrent layers
nonlinearity = param4  # Activation function ('tanh' or 'relu')
bias = param5  # Whether to use bias
batch_first = param6  # If True, then the input and output tensors are provided as (batch, seq, feature)
dropout = param7  # Dropout probability
bidirectional = param8  # If True, becomes a bidirectional RNN

# Create input tensor
seq_len = param9  # Length of the input sequences
batch_size = param10  # Number of samples in a batch
input_tensor = torch.randn(batch_size, seq_len, input_size)  # Shape according to batch_first

# Move tensor to GPU
input_tensor_gpu = input_tensor.cuda()

# Instantiate RNNs
rnn_cpu = nn.RNN(input_size, hidden_size, num_layers, nonlinearity, bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
rnn_gpu = nn.RNN(input_size, hidden_size, num_layers, nonlinearity, bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional).cuda()

# Forward pass on CPU and GPU
cpu_output, _ = rnn_cpu(input_tensor)  # on CPU
gpu_output, _ = rnn_gpu(input_tensor_gpu)  # on GPU

num_of_parameters = 10

# Parameters:
#   - param1: Size of input features : Range = [1, 1024] : Type = int
#   - param2: Number of features in the hidden state : Range = [1, 1024] : Type = int
#   - param3: Number of recurrent layers : Range = [1, 10] : Type = int
#   - param4: Nonlinearity type : Range = ['tanh', 'relu'] : Type = str
#   - param5: Use bias : Range = [True, False] : Type = bool
#   - param6: Batch first : Range = [True, False] : Type = bool
#   - param7: Dropout : Range = [0.0, 1.0] : Type = float
#   - param8: Bidirectional : Range = [True, False] : Type = bool
#   - param9: Length of the input sequences : Range = [1, 100] : Type = int
#   - param10: Number of samples in a batch : Range = [1, 100] : Type = intparam1 = int(450)
param2 = int(700)
param3 = int(9)
param4 = str('relu')
param5 = bool(False)
param6 = bool(True)
param7 = float(0.15)
param8 = bool(False)
param9 = int(85)
param10 = int(38)
