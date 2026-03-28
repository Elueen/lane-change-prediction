from NGSIM_env.envs.ngsim_env import NGSIMEnv

period = 0
counter = []

for i in range(3200):
    try:
        vehicle_id = i
        env = NGSIMEnv(scene='us-101', period=period, vehicle_id=vehicle_id, IDM=False)
        counter.append([i, env.vehicle.ngsim_traj.shape[0]])
    except:
        continue

with open('test_counter', 'w') as file:
    for item in counter:
        file.write(f'{item[0]} {item[1]}\n')
