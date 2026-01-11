import torch
import torch.nn as nn

# Classification problem
class Model(nn.Module):
    def __init__(self, num_variables, num_locations, num_hiddens, batch_size, forecast_steps=1):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=num_variables*num_locations, hidden_size=num_hiddens, batch_first=True)
        # self.lstm = nn.LSTM(input_size=4, hidden_size=num_hiddens, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.LazyLinear(out_features=forecast_steps)
        )
        

    def forward(self, x):
        h0 = torch.zeros((1, self.batch_size, self.num_hiddens))
        c0 = torch.zeros((1, self.batch_size, self.num_hiddens))
        lstm_output, (h_n, c_n) = self.lstm(x, (h0, c0))
        # Since num_layers = 1, lstm_output shape is (batch_size, lookback, num_hiddens)
        output = self.feedforward(lstm_output[:, -1, :])
        return output
