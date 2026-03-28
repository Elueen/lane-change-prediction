import torch
from torch import nn


class GRUNet1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUNet1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):

        output, hidden = self.gru(x, hidden)
        output = self.fc(output[:, -1:, :])
        return output, hidden


class GRUNet2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUNet2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):

        _, h_n = self.gru(x)
        h_n = h_n.view(-1, self.fc.in_features)
        output = self.fc(h_n)
        return output


if __name__ == "__main__":
    model = GRUNet(input_size=100, hidden_size=20, output_size=10, num_layers=1)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(total_params))
