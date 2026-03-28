# from Models.models import RNNnet, LSTMnet, GRUnet, LPRNN, LPLSTM, LPGRU
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
from imblearn.over_sampling import RandomOverSampler
import re


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

    tensor1 = torch.from_numpy(X_resampled).view(-1, 25, 9)
    tensor2 = torch.from_numpy(y_resampled).view(-1)

    re_tensor = TensorDataset(tensor1, tensor2)

    return re_tensor


def dismantle_data(model_name, file_path, mfi, batch_size):

    tensor = torch.load(file_path)
    tensor[:, :, -1] = torch.where(tensor[:, :, -1] > 6, torch.tensor(6.0), tensor[:, :, -1])

    train_size = int(0.7 * len(tensor))
    val_size = int(0.15 * len(tensor))
    test_size = len(tensor) - train_size - val_size

    if model_name in ["RNN", "LSTM", "GRU"]:
        labels = tensor[:, -1, -2].clone()
    else:
        labels = cl_labels(tensor)

    norm_tensor = F.normalize(tensor, p=2, dim=-1)
    tensor_dataset = TensorDataset(norm_tensor[:, :-10, :], labels)
    training_tensor, validating_tensor, testing_tensor = random_split(tensor_dataset, [train_size,
                                                                                       val_size,
                                                                                       test_size])

    if model_name in ["LPRNN", "LPLSTM", "LPGRU"]:
        if mfi == "resampling":
            training_tensor = oversampling_16(training_tensor)
            weights = 114514
        else:
            _, training_labels = training_tensor.dataset[training_tensor.indices]
            weights = compute_class_weights(training_labels)
            pass
    else:
        weights = 114514
        pass

    training_set = DataLoader(training_tensor, batch_size=batch_size, shuffle=True)
    validating_set = DataLoader(validating_tensor, batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(testing_tensor, batch_size=batch_size, shuffle=False)

    return training_set, validating_set, testing_set, weights


# def get_model(model_name, hidden_size, num_layers):
#     model_classes = {
#         "RNN": RNN,
#         "LSTM": LSTM,
#         "GRU": GRU,
#         "LPRNN": LPRNN,
#         "LPLSTM": LPLSTM,
#         "LPGRU": LPGRU
#     }
#
#     if model_name in model_classes:
#         return model_classes[model_name](input_size=9, hidden_size=hidden_size, output_size=1 if model_name in ["RNN", "LSTM", "GRU"] else 3, num_layers=num_layers)
#     else:
#         raise ValueError(f"Unsupported model_name: {model_name}")


def get_loss_fn(model_name, mfi, weights):
    if model_name in ["LPRNN", "LPLSTM", "LPGRU"]:
        if mfi == "resampling":
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.MSELoss()

    return loss_fn


def get_hidden(model_name, num_layers, batch_size, hidden_size):
    if model_name in ["LSTM", "LPLSTM"]:
        h_state = nn.init.normal_(torch.empty(num_layers, batch_size, hidden_size).float(), std=0.015)
        c_state = nn.init.normal_(torch.empty(num_layers, batch_size, hidden_size).float(), std=0.015)
        hidden = (h_state, c_state)
    else:
        hidden = torch.zeros(num_layers, batch_size, hidden_size)

    return hidden


def get_number(str):
    numbers = re.findall(r'\d+', str)

    return int(numbers[0])


def get_lane(tensor):
    width = 12 / 3.281
    labels = (tensor / width + 1.5).int()

    return labels


def get_cl_labels(tensor1, tensor2):

    labels = torch.where(tensor1 < tensor2, 0, torch.where(tensor1 == tensor2, 1, 2))

    return labels
