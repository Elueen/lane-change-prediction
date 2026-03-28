from Models.utils import *
from Models.arg import parse_arguments
from Models.models import RNNModel
import torch
from itertools import product
import logging
from argparse import Namespace


params = parse_arguments()
logging.basicConfig(filename="logs/7_25_320.txt", level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params["device"] = device

param_grid = {
    "model_name": ["LPRNN", "LPLSTM", "LPGRU"],
    "batch_size": [16, 32, 64],
    "hidden_size": [16, 32, 64, 128, 256],
    "num_layers": [1, 2],
    "lr": [0.0001, 0.001, 0.01]
}

param_combinations = product(*param_grid.values())

best_model_params = None
best_model_score = float('-inf')

for param in param_combinations:
    current_model_params = dict(zip(param_grid.keys(), param))
    params.update(current_model_params)

    N_params = Namespace(**params)

    logging.basicConfig(filename="logs/114514.txt", level=logging.INFO)

    training_set, validating_set, testing_set, weights = dismantle_data(N_params .model_name,
                                                                        N_params .file_path,
                                                                        N_params .mfi,
                                                                        N_params .batch_size)

    model = RNNModel(N_params, logging)
    model.fit(training_set, validating_set)
    accuracy = model.get_accuracy()

    if accuracy > best_model_score:
        best_model_score = accuracy
        best_model_params = current_model_params

logging.info("Best model score: {}".format(best_model_score))
logging.info("Best model params: {}".format(best_model_params))
