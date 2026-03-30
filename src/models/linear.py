import torch
import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):

        output = self.linear(x)

        return output
