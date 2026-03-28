import torch

model = torch.load("Models/save/test_model01.pth")
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters: {}".format(total_params))
