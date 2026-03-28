import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import time
from Models.LSTM import LSTMNet
from Models.LPLSTM import LPLSTM
from Models.utils import cl_labels, compute_class_weights
import pdb

torch.autograd.set_detect_anomaly(True)

n_epochs = 1000
learning_rate = 1e-4
batch_size = 16
num_layers_re = 2
num_layers_cl = 1
hidden_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename="logs/7_61_320.txt", level=logging.INFO)
logging.info("Device: {}".format(device))

tensor = torch.load("outcome/files/CCG_features320.pth")
tensor[:, :, -1] = torch.where(tensor[:, :, -1] > 6, torch.tensor(6.0), tensor[:, :, -1])

# train_size = int(0.7 * len(tensor))
# val_size = int(0.15 * len(tensor))
# test_size = len(tensor) - train_size - val_size
splitting_list = [0.3, 0.3, 0.15, 0.05, 0.05, 0.05, 0.1]

labels = cl_labels(tensor)
weights = compute_class_weights(labels)
sequence_size = tensor[:, :-10, :].size(1)
tensor_dataset = TensorDataset(tensor[:, :-10, :], labels)
tra_re, tra_cl, tra_ln, val_re, val_cl, val_ln, testing_tensor = random_split(tensor_dataset, splitting_list)

training_set_re = DataLoader(tra_re, batch_size=batch_size, shuffle=True)
training_set_cl = DataLoader(tra_cl, batch_size=batch_size, shuffle=True)
training_set_ln = DataLoader(tra_ln, batch_size=batch_size, shuffle=True)
validating_re = DataLoader(val_re, batch_size=batch_size, shuffle=True)
validating_cl = DataLoader(val_cl, batch_size=batch_size, shuffle=True)
validating_ln = DataLoader(val_ln, batch_size=batch_size, shuffle=True)
testing_set = DataLoader(testing_tensor, batch_size=batch_size, shuffle=False)

model_re = LPLSTM(input_size=9, hidden_size=64, output_size=1, num_layers=num_layers_re, sequence_size=sequence_size)
model_re = model_re.to(device)

model_cl = LPLSTM(input_size=9, hidden_size=64, output_size=3, num_layers=num_layers_cl, sequence_size=sequence_size)
model_cl = model_cl.to(device)

model_ln = nn.Linear(2, 1)
model_ln = model_ln.to(device)

loss_fn_re = nn.CrossEntropyLoss(weight=weights)
loss_fn_re = loss_fn_re.to(device)

loss_fn_re = nn.CrossEntropyLoss(weight=weights)
loss_fn_re = loss_fn_re.to(device)

optimizer = torch.optim.Adam(model_re.parameters(), lr=learning_rate_re)
optimizer = torch.optim.Adam(model_re.parameters(), lr=learning_rate_re)
optimizer = torch.optim.Adam(model_re.parameters(), lr=learning_rate_re)

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

    if epoch >= 50:
        if no_improvement_count >= early_stopping_patience:
            logging.info("Epoch: {}/{}, Train Loss: {}, Best accuracy: {}".format(epoch + 1, n_epochs, total_loss, best_accuracy))
            break

    if (epoch+1) % 10 == 0:
        logging.info("Epoch: {}/{}, Train Loss: {}, Best accuracy: {}".format(epoch + 1, n_epochs, total_loss, best_accuracy))

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

    logging.info("Learning Rate: {}".format(learning_rate))
    logging.info("Number of layers: {}".format(num_layers))
    logging.info("Accuracy: {}".format(accuracy))
    logging.info("Precision: {}".format(precision))
    logging.info("Recall: {}".format(recall))
    logging.info("F1: {}".format(f1))

    class_names = ["Turn left", "Idle", "Turn right"]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("outcome/plots/7_61_320.png")

