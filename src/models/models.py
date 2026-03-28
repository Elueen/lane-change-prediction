import torch
from torch import nn
from sklearn.metrics import accuracy_score
import logging
from Models.utils import get_number


class RNNModel(nn.Module):
    def __init__(self, params=None, logger=None):
        super(RNNModel, self).__init__()
        self.params = params
        self.logger = logger

        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Logger is None. Using default logger.")

        if params is not None:
            params_dict = vars(params)
            for (arg, value) in params_dict.items():
                self.logger.info("Argument %s: %r", arg, value)
        else:
            self.logger.warning("Params is None. Model initialized with default parameters.")

        self.cell_type = params.model_name
        self.file_path = params.file_path
        self.s_num = get_number(self.file_path)
        self.device_type = params.device
        self.mfi = params.mfi
        self.learning_rate = params.lr
        self.n_epochs = params.num_epoch
        self.min_epochs = params.min_epochs
        self.early_stop = params.early_stop
        self.batch_size = params.batch_size
        self.input_size = 9
        self.hidden_size = params.hidden_size
        self.output_size = 1 if self.cell_type in ["RNN", "LSTM", "GRU"] else 3
        self.dropout = 0.0
        self.num_layers = params.num_layers
        self.sequence_size = 25
        self.best_val_loss = float("inf")
        self.best_accuracy = 0
        if self.cell_type == "RNN":
            self.model = nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                batch_first=True)
        if self.cell_type == "LSTM":
            self.model = nn.LSTM(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout,
                                 batch_first=True)
        if self.cell_type == "GRU":
            self.model = nn.GRU(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                batch_first=True)
        if self.cell_type == "LPRNN":
            self.model = nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                batch_first=True)
        if self.cell_type == "LPLSTM":
            self.model = nn.LSTM(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout,
                                 batch_first=True)
        if self.cell_type == "LPGRU":
            self.model = nn.GRU(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                batch_first=True)
        self.batch_norm = nn.BatchNorm1d(self.sequence_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        if self.cell_type in ["RNN", "LSTM", "GRU"]:
            self.loss_fn = nn.MSELoss()
        elif self.cell_type in ["LPRNN", "LPLSTM", "LPGRU"]:
            self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        if self.cell_type in ["RNN", "GRU", "LPRNN", "LPGRU"]:
            output, _ = self.model(x)
            # output = h_n.view(-1, self.fc.in_features)
            output = self.batch_norm(output)

        elif self.cell_type in ["LSTM", "LPLSTM"]:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            output, _ = self.model(x, (h0, c0))
            output = self.batch_norm(output)

        output = self.fc(output[:, -1, :])

        if self.cell_type in ["LPRNN", "LPLSTM", "LPGRU"]:
            output = self.softmax(output)

        return output

    def fit(self, training_set, validating_set, save_model=False):

        n_epochs = self.n_epochs
        device = self.device_type

        # model = self.model
        # model = model.to(device)

        loss_fn = self.loss_fn
        loss_fn = loss_fn.to(device)
        optimizer = self.optimizer
        min_epochs = self.min_epochs
        early_stopping_patience = self.early_stop
        no_improvement_count = 0
        best_val_loss = self.best_val_loss
        best_accuracy = self.best_accuracy

        for epoch in range(n_epochs):

            # model.train()

            total_loss = 0.0

            for train_inputs, train_labels in training_set:
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)
                train_outputs = self(train_inputs)

                optimizer.zero_grad()
                loss = loss_fn(train_outputs, train_labels)
                assert not torch.isnan(loss).any(), "Loss contains NaN values"
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            total_loss /= len(training_set)

            # model.eval()
            with torch.no_grad():

                val_loss = 0.0
                val_accuracy = 0.0

                for val_inputs, val_labels in validating_set:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = self(val_inputs)

                    if self.cell_type in ["RNN", "LSTM", "GRU"]:
                        v_loss = loss_fn(val_outputs, val_labels)
                        val_loss += v_loss.item()
                    elif self.cell_type in ["LPRNN", "LPLSTM", "LPGRU"]:
                        val_pred_labels = torch.argmax(val_outputs, dim=1)
                        val_accuracy += accuracy_score(val_labels.cpu().numpy(), val_pred_labels.cpu().numpy())

                    val_loss /= len(validating_set)
                    val_accuracy /= len(validating_set)

                if self.cell_type in ["RNN", "LSTM", "GRU"]:

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                elif self.cell_type in ["LPRNN", "LPLSTM", "LPGRU"]:
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

            if self.cell_type in ["RNN", "LSTM", "GRU"]:
                if epoch >= min_epochs:
                    if no_improvement_count >= early_stopping_patience:
                        self.logger.info("Epoch: {}/{}, Train Loss: {}, Val Loss: {}".format(epoch + 1,
                                                                                             n_epochs,
                                                                                             total_loss,
                                                                                             val_loss))
                        break

                if (epoch + 1) % 10 == 0:
                    self.logger.info("Epoch: {}/{}, Train Loss: {}, Val Loss: {}".format(epoch + 1,
                                                                                         n_epochs,
                                                                                         total_loss,
                                                                                         val_loss))

            elif self.cell_type in ["LPRNN", "LPLSTM", "LPGRU"]:
                if epoch >= 50:
                    if no_improvement_count >= early_stopping_patience:
                        self.logger.info("Epoch: {}/{}, Train Loss: {}, Best accuracy: {}".format(epoch + 1,
                                                                                                  n_epochs,
                                                                                                  total_loss,
                                                                                                  best_accuracy))
                        break

                if (epoch + 1) % 10 == 0:
                    self.logger.info("Epoch: {}/{}, Train Loss: {}, Best accuracy: {}".format(epoch + 1,
                                                                                              n_epochs,
                                                                                              total_loss,
                                                                                              best_accuracy))

        self.best_val_loss = best_val_loss
        self.best_accuracy = best_accuracy

        # if save_model:
        #     torch.save(model.state_dict(), "Models/save/enenen.pth")

    def get_loss(self):
        return self.best_val_loss

    def get_accuracy(self):
        return self.best_accuracy

 # def predict(self, x, using_best=False):
 #     final_lr = optimizer.param_groups[0]['lr']
 #
 #     model.eval()
 #     test_hidden = get_hidden(model_name, num_layers, batch_size, hidden_size)
 #     test_hidden = test_hidden.to(device)
 #     real_labels = []
 #     pred_labels = []
 #
 #     with torch.no_grad():
 #         for inputs, real_label in testing_set:
 #             inputs = inputs.to(device)
 #             real_label = real_label.to(device)
 #             pred_list, _ = model(inputs, test_hidden)
 #             pred_label = torch.argmax(pred_list, dim=1)
 #
 #             real_labels.extend(real_label.cpu().numpy())
 #             pred_labels.extend(pred_label.cpu().numpy())
 #
 #     accuracy = accuracy_score(real_labels, pred_labels)
 #     conf_matrix = confusion_matrix(real_labels, pred_labels)
 #
 #     logging.info("Total Epochs: {}".format(n_epochs))
 #     logging.info("Initial Learning Rate: {}".format(learning_rate))
 #     logging.info("Final Learning Rate: {}".format(final_lr))
 #     logging.info("Accuracy: {:.2f}%".format(accuracy * 100))
 #
 #     class_names = ["Turn left", "Idle", "Turn right"]
 #     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
 #     plt.xlabel("Predicted")
 #     plt.ylabel("True")
 #     plt.title("Confusion Matrix")
 #     plt.savefig("outcome/plots/{}_{}_{}_{}.png".format(model_name, s_num, mfi, learning_rate))

