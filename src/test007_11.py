from Models.utils import *
from Models.LPGRU import LPGRU
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import time
import pdb

n_epochs = 500
model_name = "LPGRU"
learning_rate = 1e-4
batch_size = 16
num_layers = 1
file_path = "outcome/files/CCG_features100.pth"
mfi = "resampling"
hidden_size = 64
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     raise RuntimeError("SBCSF")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename="logs/GRU_Cla100_X.txt", level=logging.INFO)
logging.info("Device: {}".format(device))

training_set, validating_set, testing_set, weights = dismantle_data(model_name, file_path, mfi, batch_size)

model = LPGRU(input_size=9, hidden_size=hidden_size, output_size=3, num_layers=num_layers)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
hidden_size = model.hidden_size
# hidden = torch.zeros(num_layers, batch_size, hidden_size)
# hidden = hidden.to(device)
# cell = torch.zeros(num_layers, batch_size, hidden_size)
# cell = cell.to(device)
total_time = 0
early_stopping_patience = 10
best_accuracy = 0
no_improvement_count = 0

for epoch in range(n_epochs):

    model.train()
    start_time = time.time()
    total_loss = 0.0

    train_hidden = get_hidden(model_name, num_layers, batch_size, hidden_size)
    train_hidden = train_hidden.to(device)

    for inputs, label in training_set:
        inputs = inputs.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(inputs)

        loss = loss_fn(output, label)
        assert not torch.isnan(loss).any(), "Loss contains NaN values"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    total_loss /= len(training_set)

    model.eval()
    with torch.no_grad():
        val_accuracy = 0.0
        val_hidden = get_hidden(model_name, num_layers, batch_size, hidden_size)
        val_hidden = val_hidden.to(device)
        for inputs, label in validating_set:
            inputs = inputs.to(device)
            label = label.to(device)
            output = model(inputs)

            pred_output = torch.argmax(output, dim=1)

            label.cpu().detach().numpy()
            pred_output.cpu().detach().numpy()
            val_accuracy += accuracy_score(label, pred_output)

    val_accuracy /= len(validating_set)

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    end_time = time.time()
    epoch_time = end_time - start_time
    total_time += epoch_time

    if epoch >= 200:
        if no_improvement_count >= early_stopping_patience:
            logging.info("Epoch: {}/{}, Train Loss: {}, Val acc: {}, Training time: {:.2f} sec".format(epoch + 1,
                                                                                                       n_epochs,
                                                                                                       total_loss,
                                                                                                       best_accuracy,
                                                                                                       total_time))
            break

    if (epoch+1) % 10 == 0:
        logging.info("Epoch: {}/{}, Train Loss: {}, Val acc: {}, Training time: {:.2f} sec".format(epoch + 1,
                                                                                                   n_epochs,
                                                                                                   total_loss,
                                                                                                   best_accuracy,
                                                                                                   total_time))

# torch.save(model.state_dict(), "Models/save/GRU_Cla344_500_-3.pth")
final_lr = optimizer.param_groups[0]['lr']

if __name__ == "__main__":

    model.eval()

    real_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in testing_set:
            inputs, real_label = batch
            inputs = inputs.to(device)
            real_label = real_label.to(device)
            pred_list = model(inputs)
            pred_label = torch.argmax(pred_list, dim=1)

            real_labels.extend(real_label.cpu().numpy())
            pred_labels.extend(pred_label.cpu().numpy())

    accuracy = accuracy_score(real_labels, pred_labels)
    conf_matrix = confusion_matrix(real_labels, pred_labels)

    logging.info("Total Epochs: {}".format(n_epochs))
    logging.info("Initial Learning Rate: {}".format(learning_rate))
    logging.info("Final Learning Rate: {}".format(final_lr))
    logging.info("Accuracy: {:.2f}%".format(accuracy * 100))

    class_names = ["Turn left", "Idle", "Turn right"]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("outcome/plots/GRU_Cla100_X.png")
