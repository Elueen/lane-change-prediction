from NGSIM_env.envs.ngsim_env import NGSIMEnv
import matplotlib.pyplot as plt
from NGSIM_env.utils import *
import numpy as np
import csv
from NGSIM_env.utils import *

# parameters
n_iters = 200
lr = 0.05
lam = 0.01
feature_num = 8
vehicles = [2617, 148, 809, 1904, 241, 2204, 619, 2390, 1267, 1269, 1370, 80, 1908, 820, 2293, 2218, 1014, 1221, 2489, 2284]
period = 0
render_env = False

# Simulate trajectory
for i in vehicles:
    # select vehicle
    vehicle_id = i
    print('Target Vehicle: {}'.format(vehicle_id))

    # create environment
    env = NGSIMEnv(scene='us-101', period=period, vehicle_id=vehicle_id, IDM=False)
    plt.plot(env.vehicle.ngsim_traj[:, 0] / 3.281, env.vehicle.ngsim_traj[:, 1] / 3.281)


try:
    plt.gca().set_aspect('auto', 'datalim')
    plt.gca().set(xlim=(0, 660), ylim=(0, 25))
    plt.gca().invert_yaxis()
    plt.xlabel('Longitudinal position [m]', fontsize=20)
    plt.ylabel('Lateral position [m]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig('traj_plot317.png')