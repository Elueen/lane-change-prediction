
import numpy as np
import json


def load_json_file(filename):
    with open(filename, "r") as json_file:
        return json.load(json_file)


def write_json_file(filename, files):
    with open(filename, "w") as json_file:
        return json.dump(files, json_file)


traj_list = load_json_file("outcome/files/sorted_vel_4600.json")
cl_vel_list = load_json_file("outcome/files/cl_vel_list01.json")


new_list = [value for value in traj_list if value in cl_vel_list]
filename = "outcome/files/sorted_vel_2200.json"

write_json_file(filename, new_list)
print("traj_list:", len(traj_list))
print("cl_vel_list:", len(cl_vel_list))
print("new_list:", len(new_list))
