import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from Models.LSTM import LSTMNet
import pdb

torch.autograd.set_detect_anomaly(True)

n_epochs = 300
learning_rate = 1e-4
batch_size = 16
num_layers = 1
hidden_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor = torch.load("outcome/files/CCG_features100.pth")
tensor[:, :, -1] = torch.where(tensor[:, :, -1] > 6, torch.tensor(6.0), tensor[:, :, -1])

train_size = int(0.7 * len(tensor))
val_size = int(0.15 * len(tensor))
test_size = len(tensor) - train_size - val_size

traj_tensor = tensor[:, -1, -2].clone()
sequence_size = tensor[:, :-10, :].size(1)
tensor_dataset = TensorDataset(tensor[:, :-10, :], traj_tensor)
training_tensor, validating_tensor, testing_tensor = random_split(tensor_dataset,
                                                                  [train_size, val_size, test_size])

training_set = DataLoader(training_tensor, batch_size=batch_size, shuffle=True)
validating_set = DataLoader(validating_tensor, batch_size=batch_size, shuffle=True)
testing_set = DataLoader(testing_tensor, batch_size=batch_size, shuffle=False)

model = LSTMNet(input_size=9, hidden_size=64, output_size=1, num_layers=num_layers, sequence_size=sequence_size)
model = model.to(device)

loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

early_stopping_patience = 5
best_val_loss = float("inf")
no_improvement_count = 0

for epoch in range(n_epochs):

    model.train()

    total_loss = 0.0

    for train_inputs, train_traj in training_set:
        train_inputs = train_inputs.to(device)
        train_traj = train_traj.to(device)
        train_traj = train_traj.unsqueeze(1)
        train_output = model(train_inputs)

        optimizer.zero_grad()
        loss = loss_fn(train_output, train_traj)
        assert not torch.isnan(loss).any(), "Loss contains NaN values"
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    total_loss /= len(training_set)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_inputs, val_traj in validating_set:
            val_inputs = val_inputs.to(device)
            val_traj = val_traj.to(device)
            val_traj = val_traj.unsqueeze(1)
            val_outputs = model(val_inputs)

            v_loss = loss_fn(val_outputs, val_traj)

            val_loss += v_loss.item()
    val_loss /= len(validating_set)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= early_stopping_patience:
        print("Epoch: {}/{}, Train Loss: {}, Val Loss: {}".format(epoch + 1, n_epochs, total_loss, val_loss))
        break

    if (epoch+1) % 10 == 0:
        print("Epoch: {}/{}, Train Loss: {}, Val Loss: {}".format(epoch + 1, n_epochs, total_loss, val_loss))

# torch.save(model.state_dict(), "Models/save/test_LSTM_model01.pth")


model.eval()

total_mse = 0.0
total_mae = 0.0
total_rmse = 0.0

with torch.no_grad():

    for test_inputs, test_traj in testing_set:
        test_inputs = test_inputs.to(device)
        test_traj = test_traj.to(device)
        test_traj = test_traj.unsqueeze(1)
        test_outputs = model(test_inputs)

        mse = torch.mean((test_outputs - test_traj) ** 2)
        mae = torch.mean(torch.abs(test_outputs - test_traj))
        rmse = torch.sqrt(mse)

        total_mse += mse.item()
        total_mae += mae.item()
        total_rmse += rmse.item()

    average_mse = total_mse / len(testing_set)
    average_mae = total_mae / len(testing_set)
    average_rmse = total_rmse / len(testing_set)

    print("Mean Squared Error (MSE): {}".format(average_mse))
    print("Mean Absolute Error (MAE): {}".format(average_mae))
    print("Root Mean Squared Error (RMSE): {}".format(average_rmse))
    print("Total Epochs: {}".format(n_epochs))
    print("Learning Rate: {}".format(learning_rate))
