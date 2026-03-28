import torch
from torch import nn


class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):

        output, hidden = self.rnn(x, hidden)
        output = self.fc(output[:, -1:, :])
        return output, hidden


if __name__ == "__main__":
    model = RNNNet(input_size=100, hidden_size=20, output_size=10, num_layers=1)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(total_params))
