import torch


tensor_data = torch.load("outcome/files/CCG_features344.pth")

indices = torch.randperm(tensor_data.size(0))[:1600]

random_subset = tensor_data[indices]

torch.save(random_subset, "outcome/files/CCG_features100.pth")
