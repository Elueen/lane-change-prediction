from Models.LPRNN import LPRNN, cl_labels, compute_class_weights, oversampling_16
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

n_epochs = 1000
learning_rate = 1e-3
batch_size = 16
num_layers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# pdb.set_trace()

tensor = torch.load("outcome/files/CCG_features100.pth")

# has_nan = torch.isnan(tensor).any().item()
# print("Data contains NaN values:", has_nan)  # False
# pdb.set_trace()
train_size = int(0.7 * len(tensor))
test_size = len(tensor) - train_size
mean = tensor.mean()
std = tensor.std()

labels = cl_labels(tensor)
# weights = compute_class_weights(labels)
norm_tensor = F.normalize(tensor, p=2, dim=-1)
tensor_dataset = TensorDataset(norm_tensor[:, :-10, :], labels)
training_tensor, testing_tensor = random_split(tensor_dataset, [train_size, test_size])
training_tensor = oversampling_16(training_tensor)


training_set = DataLoader(training_tensor, batch_size=batch_size, shuffle=True)
testing_set = DataLoader(testing_tensor, batch_size=batch_size, shuffle=False)

model = LPRNN(input_size=9, hidden_size=64, output_size=3, num_layers=num_layers)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.CrossEntropyLoss(weight=weights)
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
hidden_size = model.hidden_size


for epoch in range(n_epochs):

    model.train()

    total_loss = 0.0

    for batch in training_set:

        inputs, label = batch
        inputs = inputs.to(device)
        label = label.to(device)
        hidden = nn.init.normal_(torch.empty(num_layers, batch_size, hidden_size).float().to(device), std=0.015)
        output, hidden = model(inputs, hidden)

        loss = loss_fn(output, label)
        optimizer.zero_grad()
        # print(loss)
        assert not torch.isnan(loss).any(), "Loss contains NaN values"
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # print(total_loss)
        # pdb.set_trace()

    if (epoch+1) % 100 == 0:
        print("Iteration: {}, Loss: {}".format(epoch+1, total_loss / len(training_set)))
        # pdb.set_trace()

# torch.save(model.state_dict(), "Models/save/test_model01.pth")
final_lr = optimizer.param_groups[0]['lr']

if __name__ == "__main__":

    model.eval()

    initial_hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    real_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in testing_set:
            inputs, real_label = batch
            inputs = inputs.to(device)
            real_label = real_label.to(device)
            pred_list, _ = model(inputs, initial_hidden)
            pred_label = torch.argmax(pred_list, dim=1)

            real_labels.extend(real_label.cpu().numpy())
            pred_labels.extend(pred_label.cpu().numpy())

    accuracy = accuracy_score(real_labels, pred_labels)
    conf_matrix = confusion_matrix(real_labels, pred_labels)

    print("Total Epochs: {}".format(n_epochs))
    print("Initial Learning Rate: {}".format(learning_rate))
    print("Final Learning Rate: {}".format(final_lr))
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Confusion Matrix:")
    print(conf_matrix)
    class_names = ["Turn left", "Idle", "Turn right"]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
