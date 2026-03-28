import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Models.LPRNN import LPRNN
from Models.utils import cl_labels, compute_class_weights, oversampling_16
import pdb

torch.autograd.set_detect_anomaly(True)

n_epochs = 1000
learning_rate = 1e-2
batch_size = 16
num_layers = 1
hidden_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor = torch.load("outcome/files/CCG_features100.pth")
tensor[:, :, -1] = torch.where(tensor[:, :, -1] > 6, torch.tensor(6.0), tensor[:, :, -1])

train_size = int(0.7 * len(tensor))
val_size = int(0.15 * len(tensor))
test_size = len(tensor) - train_size - val_size

labels = cl_labels(tensor)
# weights = compute_class_weights(labels)
sequence_size = tensor[:, :-10, :].size(1)
tensor_dataset = TensorDataset(tensor[:, :-10, :], labels)
training_tensor, validating_tensor, testing_tensor = random_split(tensor_dataset,
                                                                  [train_size, val_size, test_size])
training_tensor = oversampling_16(training_tensor)

training_set = DataLoader(training_tensor, batch_size=batch_size, shuffle=True)
validating_set = DataLoader(validating_tensor, batch_size=batch_size, shuffle=True)
testing_set = DataLoader(testing_tensor, batch_size=batch_size, shuffle=False)

model = LPRNN(input_size=9, hidden_size=64, output_size=3, num_layers=num_layers, sequence_size=sequence_size)
model = model.to(device)


loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

early_stopping_patience = 5
best_accuracy = 0
no_improvement_count = 0

for epoch in range(n_epochs):

    model.train()

    total_loss = 0.0

    for train_inputs, train_labels in training_set:
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)
        train_outputs = model(train_inputs)

        optimizer.zero_grad()
        loss = loss_fn(train_outputs, train_labels)
        assert not torch.isnan(loss).any(), "Loss contains NaN values"
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    total_loss /= len(training_set)

    model.eval()
    with torch.no_grad():
        val_accuracy = 0.0
        for val_inputs, val_labels in validating_set:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            val_pred_labels = torch.argmax(val_outputs, dim=1)

            val_accuracy += accuracy_score(val_labels.cpu().numpy(), val_pred_labels.cpu().numpy())
        val_accuracy /= len(validating_set)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    if epoch >= 100:
        if no_improvement_count >= early_stopping_patience:
            print("Epoch: {}/{}, Train Loss: {}, Best accuracy: {}".format(epoch + 1, n_epochs, total_loss, best_accuracy))
            break

    if (epoch+1) % 10 == 0:
        print("Epoch: {}/{}, Train Loss: {}, Best accuracy: {}".format(epoch + 1, n_epochs, total_loss, best_accuracy))

# torch.save(model.state_dict(), "Models/save/test_LSTM_model01.pth")


model.eval()

test_real_labels = []
test_pred_labels = []

with torch.no_grad():

    for test_inputs, test_labels in testing_set:
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_inputs)
        test_pred_label = torch.argmax(test_outputs, dim=1)

        test_real_labels.extend(test_labels.cpu().numpy())
        test_pred_labels.extend(test_pred_label.cpu().numpy())

    accuracy = accuracy_score(test_real_labels, test_pred_labels)
    precision = precision_score(test_real_labels, test_pred_labels, average="macro")
    recall = recall_score(test_real_labels, test_pred_labels, average="macro")
    f1 = f1_score(test_real_labels, test_pred_labels, average="macro")
    conf_matrix = confusion_matrix(test_real_labels, test_pred_labels)

    print("Learning Rate: {}".format(learning_rate))
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

