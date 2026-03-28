import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from Models.GRUNet import GRUNet1, GRUNet2
import pdb

n_epochs = 100
learning_rate = 1e-3
batch_size = 16
num_layers = 1
hidden_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# pdb.set_trace()

tensor = torch.load("outcome/files/CCG_features100.pth")
tensor[:, :, -1] = torch.where(tensor[:, :, -1] > 6, torch.tensor(6.0), tensor[:, :, -1])
# has_nan = torch.isnan(tensor).any().item()
# print("Data contains NaN values:", has_nan)  # False
# pdb.set_trace()

train_size = int(0.7 * len(tensor))
test_size = len(tensor) - train_size
mean = tensor.mean()
std = tensor.std()

traj_tensor = tensor[:, -1, -2].clone()
norm_tensor = F.normalize(tensor, p=2, dim=-1)
tensor_dataset = TensorDataset(norm_tensor[:, :-10, :], traj_tensor)
training_tensor, testing_tensor = random_split(tensor_dataset, [train_size, test_size])

training_set = DataLoader(training_tensor, batch_size=batch_size, shuffle=True)
testing_set = DataLoader(testing_tensor, batch_size=batch_size, shuffle=False)

model = GRUNet1(input_size=9, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
model = model.to(device)

loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
hidden_size = model.hidden_size
# hidden = torch.zeros(num_layers, batch_size, hidden_size)
# hidden = hidden.to(device)
# cell = torch.zeros(num_layers, batch_size, hidden_size)
# cell = cell.to(device)

for epoch in range(n_epochs):

    model.train()

    total_loss = 0.0
    hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    for batch in training_set:

        inputs, traj = batch
        inputs = inputs.to(device)
        traj = traj.to(device)
        traj = traj.unsqueeze(1).unsqueeze(1)
        output, hidden = model(inputs, hidden)

        loss = loss_fn(output, traj)
        optimizer.zero_grad()
        # print(loss)
        assert not torch.isnan(loss).any(), "Loss contains NaN values"
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # print(total_loss)
        # pdb.set_trace()

    if (epoch+1) % 10 == 0:
        print("Iteration: {}, Loss: {}".format(epoch+1, total_loss / len(training_set)))
        # pdb.set_trace()

torch.save(model.state_dict(), "Models/save/test_model01.pth")

if __name__ == "__main__":

    model.eval()
    initial_hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)

    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in testing_set:
            inputs, real_traj = batch
            inputs = inputs.to(device)
            real_traj = real_traj.to(device)
            pred_traj, _ = model(inputs, initial_hidden)

            mse = torch.mean((real_traj - pred_traj) ** 2)
            mae = torch.mean(torch.abs(real_traj - pred_traj))
            rmse = torch.sqrt(mse)

            total_mse += mse.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
            total_rmse += rmse.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_mse = total_mse / total_samples
    average_mae = total_mae / total_samples
    average_rmse = total_rmse / total_samples

    print("Mean Squared Error (MSE): {}".format(average_mse))
    print("Mean Absolute Error (MAE): {}".format(average_mae))
    print("Root Mean Squared Error (RMSE): {}".format(average_rmse))
    print("Total Samples: {}".format(total_samples))
    print("Total Epochs: {}".format(n_epochs))
    print("Learning Rate: {}".format(learning_rate))
