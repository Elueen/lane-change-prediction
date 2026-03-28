from NGSIM_env.envs.ngsim_env import NGSIMEnv
from NGSIM_env.utils import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

n_iters = 200
lr = 0.05
lam = 0.01
feature_num = 8
period = 0
render_env = False
vehicles = [2617, 148, 809, 1904, 241, 2204, 619, 2390, 1267, 1269, 1370, 80, 1908, 820, 2293, 2218, 1014, 1221, 2489, 2284]
input_size
hidden_size
output_size
epochs

class deepIRL():
    def __init__(self, input_size, hidden_size, output_size):

