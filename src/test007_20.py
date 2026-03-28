import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Models.LSTM import LSTMNet
from Models.utils import cl_labels, get_lane, get_cl_labels
import pdb

torch.autograd.set_detect_anomaly(True)

n_epochs = 1000
learning_rate = 1e-4
batch_size = 16
num_layers = 2
hidden_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor = torch.load("outcome/files/CCG_features320.pth")
tensor[:, :, -1] = torch.where(tensor[:, :, -1] > 6, torch.tensor(6.0), tensor[:, :, -1])

train_size = int(0.7 * len(tensor))
val_size = int(0.15 * len(tensor))
test_size = len(tensor) - train_size - val_size

traj_tensor = tensor[:, -1, -2].clone()
# lane_tensor = tensor[:, -11, -1].clone()
labels = cl_labels(tensor)
sequence_size = tensor[:, :-10, :].size(1)
tensor_dataset = TensorDataset(tensor[:, :-10, :], traj_tensor, labels)
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

    for train_inputs, train_traj, _ in training_set:
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
        for val_inputs, val_traj, _ in validating_set:
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

torch.save(model.state_dict(), "Models/save/1919810.pth")


model.eval()

test_real_labels = []
test_pred_labels = []

with torch.no_grad():

    for test_inputs, test_traj, test_real_label in testing_set:
        test_inputs = test_inputs.to(device)
        test_traj = test_traj.to(device)
        test_outputs = model(test_inputs)
        test_pred_lane = get_lane(test_outputs.squeeze())
        test_lane = test_inputs[:, -1, -1]
        test_pred_label = get_cl_labels(test_lane, test_pred_lane)

        test_real_labels.extend(test_real_label.cpu().numpy())
        test_pred_labels.extend(test_pred_label.cpu().numpy())

    accuracy = accuracy_score(test_real_labels, test_pred_labels)
    precision = precision_score(test_real_labels, test_pred_labels, average="macro")
    recall = recall_score(test_real_labels, test_pred_labels, average="macro")
    f1 = f1_score(test_real_labels, test_pred_labels, average="macro")
    conf_matrix = confusion_matrix(test_real_labels, test_pred_labels)

    print("Learning Rate: {}".format(learning_rate))
    print("Number of layers: {}".format(num_layers))
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))

    class_names = ["Turn left", "Idle", "Turn right"]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

