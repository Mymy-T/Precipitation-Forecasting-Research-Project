import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_variables, num_locations, num_hiddens, forecast_steps, batch_size):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=num_variables*num_locations, hidden_size=num_hiddens, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.LazyLinear(out_features=forecast_steps),
            nn.ReLU()
        )
        

    def forward(self, x):
        h0 = torch.zeros((1, self.batch_size, self.num_hiddens))
        c0 = torch.zeros((1, self.batch_size, self.num_hiddens))

        _, (h_out, _) = self.lstm(x, (h0, c0))
        return self.feedforward(h_out[0])

# test
# batch = 2
# input = torch.rand((batch, 24, 16*5))
# model = Model(5, 16, 20, 6, batch)
# print(model(input))