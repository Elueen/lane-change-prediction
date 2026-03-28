import torch
from torch import nn
from torch.utils.data import TensorDataset
from imblearn.over_sampling import RandomOverSampler


class LPRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, sequence_size):
        super(LPRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.sequence_size = sequence_size
        self.rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)
        self.batch_norm = nn.BatchNorm1d(self.sequence_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        _, h_n = self.rnn(x)
        h_n = h_n.view(-1, self.fc.in_features)
        output = self.fc(h_n)
        output = self.softmax(output)
        return output
        # output, hidden = self.rnn(x, hidden)
        # output = self.fc(output[:, -1, :])
        # output = self.softmax(output)
        # return output, hidden


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


def oversampling_16(tensor):
    training_data, training_labels = tensor.dataset[tensor.indices]
    ros = RandomOverSampler(random_state=42)
    training_data_2d = training_data.view(training_data.shape[0], -1)
    X_resampled, y_resampled = ros.fit_resample(training_data_2d, training_labels)
    remainder = len(X_resampled) % 16

    if remainder != 0:
        X_resampled = X_resampled[:-remainder]
        y_resampled = y_resampled[:-remainder]

    tensor1 = torch.from_numpy(X_resampled).view(-1, 20, 9)
    tensor2 = torch.from_numpy(y_resampled).view(-1)

    re_tensor = TensorDataset(tensor1, tensor2)

    return re_tensor


if __name__ == "__main__":
    model = LPRNN(input_size=100, hidden_size=20, output_size=10, num_layers=1)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(total_params))
