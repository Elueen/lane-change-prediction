import torch
from torch import nn


class LPLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, sequence_size):
        super(LPLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=0.5)
        self.sequence_size = sequence_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.batch_norm = nn.BatchNorm1d(self.sequence_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        output = self.dropout(output)
        output = self.batch_norm(output)
        output = self.fc(output[:, -1, :])
        output = self.softmax(output)

        return output


def cl_labels(tensor):

    labels = torch.zeros(tensor.shape[0], dtype=torch.long)

    for i in range(tensor.shape[0]):
        if tensor[i, -11, -1] < tensor[i, -1, -1]:
            labels[i] = 0
        elif tensor[i, -11, -1] == tensor[i, -1, -1]:
            labels[i] = 1
        else:
            labels[i] = 2

    return labels


def compute_class_weights(labels):
    class_counts = torch.bincount(labels)
    total_samples = len(labels)

    weights = total_samples / (class_counts.float() + 1e-10)

    weights = weights / weights.sum()

    return weights


if __name__ == "__main__":
    model = LPLSTM(input_size=100, hidden_size=20, output_size=10, num_layers=1)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(total_params))
