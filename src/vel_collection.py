import torch

original_tensor = torch.load("outcome/files/CCG_features344.pth")

random_indices = torch.randperm(original_tensor.size(0))[:1600]

subset_tensor = original_tensor[random_indices]

torch.save(subset_tensor, "outcome/files/CCG_features100.pth")
