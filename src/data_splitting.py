import torch
import json
import pdb


with open("outcome/files/LSTM_features5.json", "r") as json_file:
    json_set = json.load(json_file)

tensor = torch.tensor(json_set, dtype=torch.float32)

for i in range(tensor.shape[0]):

    column_to_sort = tensor[i, :, -2]
    sorted_indices = torch.argsort(column_to_sort)
    tensor[i, :, :] = tensor[i, sorted_indices, :]

t_tensor = torch.empty(0, 30, 10)

start_step = 0
length = 30

for i in range(tensor.shape[0]):

    step = 0
    length = 30
    t_list = []

    for j in range(21):
        s_tensor = tensor[i, step:step + length, :]
        t_list.append(s_tensor)

    stacked_tensor = torch.stack(t_list, dim=0)
    t_tensor = torch.cat((t_tensor, stacked_tensor), dim=0)

torch.save(t_tensor, "outcome/files/LSTM_features5_21.json")
