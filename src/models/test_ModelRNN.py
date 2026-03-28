import numpy as np
import torch


def get_sin_data(num_time_steps):
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    test_data = np.sin(time_steps)
    re_data = test_data.reshape(num_time_steps, 1)
    x, y = diff_data(re_data, num_time_steps)
    return x, y


def diff_data(data, time_steps):
    data = torch.tensor(data, dtype=torch.float32)
    x = data[:-1].unsqueeze(0).view(1, time_steps-1, 1)
    y = data[1:].unsqueeze(0).view(1, time_steps-1, 1)
    return x, y


if __name__ == "__main__":
    from Models.RNNNet import RNNNet

    n_iter = 50
    num_splitting = 11

    model = RNNNet(input_size=100, hidden_size=20, output_size=1, num_layers=1)
    criterion = model.loss_fn
    optimizer = model.optimizer
    hidden_size = model.hidden_size

    for i in range(n_iter):

        _x, _y = get_sin_data(num_splitting)
        hidden = torch.zeros(1, 1, hidden_size)
        output, hidden = model(_x, hidden)
        hidden = hidden.detach()

        loss = criterion(output, _y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Iteration: {}, Loss: {}".format(iter, loss.item()))
