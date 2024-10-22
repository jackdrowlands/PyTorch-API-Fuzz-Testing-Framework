import torch
param1 = 1024
param2 = 2048
param3 = 100
param4 = 4096
param5 = 10
param6 = 50
import torch
import torch.nn as nn

# Define parameters
num_of_parameters = 6

# Define the RNN and LSTM models
rnn = nn.RNN(input_size=param1, hidden_size=param2, num_layers=param3, batch_first=True)
lstm = nn.LSTM(input_size=param1, hidden_size=param4, num_layers=param5, batch_first=True)

# Generate random input
input_data = torch.randn(param6, param1)

# Stack the input data using torch.stack
stacked_input = torch.stack([input_data, input_data])

# Test RNN and LSTM on CPU
rnn_output, _ = rnn(stacked_input)
lstm_output, _ = lstm(stacked_input)
cpu_output = torch.stack([rnn_output[-1], lstm_output[-1]])

# Move models and input data to GPU
rnn.cuda()
lstm.cuda()
stacked_input = stacked_input.cuda()

# Test RNN and LSTM on GPU
rnn_output, _ = rnn(stacked_input)
lstm_output, _ = lstm(stacked_input)
gpu_output = torch.stack([rnn_output[-1], lstm_output[-1]])
print(json.dumps({'cpu_output': cpu_output.tolist(), 'gpu_output': gpu_output.cpu().tolist()}))