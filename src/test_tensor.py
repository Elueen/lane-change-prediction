
import torch
import numpy as np
import numexpr
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import json


def get_smoothed_x_y(_tensor, _window):

    smoothed_x = savgol_filter(_tensor[:, :, -2], _window, 1)
    smoothed_y = savgol_filter(_tensor[:, :, -1], _window, 1)
    smoothed_v = savgol_filter(_tensor[:, :, 0], _window, 3)
    smoothed_a1 = savgol_filter(_tensor[:, :, 1], _window, 3)
    smoothed_a2 = savgol_filter(_tensor[:, :, 2], _window, 3)
    smoothed_j = savgol_filter(_tensor[:, :, 3], _window, 3)

    return smoothed_x, smoothed_y, smoothed_v, smoothed_a1, smoothed_a2, smoothed_j


def get_smoothed_vel_accel(smoothed_x_values, smoothed_y_values, time_values, initial_vel, initial_accel):

    x_y_matrix_A = np.column_stack((smoothed_x_values, smoothed_y_values))
    x_y_matrix_B = x_y_matrix_A[1:, :]
    x_y_matrix_A = x_y_matrix_A[0:-1, :]

    dist_temp = numexpr.evaluate("sum((x_y_matrix_B - x_y_matrix_A)**2, 1)")
    dist = numexpr.evaluate("sqrt(dist_temp)")

    t_matrix_A = time_values
    t_matrix_B = t_matrix_A[1:]
    # remove last row
    t_matrix_A = t_matrix_A[0:-1]

    vel = numexpr.evaluate("dist * 1000/ (t_matrix_B - t_matrix_A)")
    smoothed_velocities = np.insert(vel, 0, initial_vel, axis=0)

    vel_matrix_A = smoothed_velocities
    vel_matrix_B = vel_matrix_A[1:]
    vel_matrix_A = vel_matrix_A[0:-1]

    acc = numexpr.evaluate("(vel_matrix_B - vel_matrix_A) * 1000/ (t_matrix_B - t_matrix_A)")
    smoothed_accelaration = np.insert(acc, 0, initial_accel, axis=0)

    return np.array(smoothed_velocities), np.array(smoothed_accelaration)


with open("outcome/files/CNT_features10.json", "r") as json_file:
    json_set = json.load(json_file)

tensor = torch.tensor(json_set, dtype=torch.float32)

window = 11
x, y, v, a1, a2, j = get_smoothed_x_y(tensor, window)

x0 = tensor[0, :, -2]
y0 = tensor[0, :, -1]


plt.subplot(2, 1, 1)
plt.plot(x0, y0)

plt.subplot(2, 1, 2)
plt.plot(x[0], y[0])

plt.show()
