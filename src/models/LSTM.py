import torch
from torch import nn
# from torch.nn.utils.rnn import PackedSequence


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, sequence_size):
        super(LSTMNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.sequence_size = sequence_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.batch_norm = nn.BatchNorm1d(self.sequence_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):

        # output, hidden = self.lstm(x, hidden)
        # output = self.fc(output[:, -1:, :])
        # return output, hidden

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        output = self.batch_norm(output)
        output = self.fc(output[:, -1, :])

        return output


if __name__ == "__main__":
    model = LSTMNet(input_size=100, hidden_size=20, output_size=10, num_layers=1)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(total_params))
