import subprocess


# commands = [
#     "python dump_data.py NGSIM_supporting_data_0.csv",
#     "python dump_data.py NGSIM_supporting_data_1.csv",
#     "python dump_data.py NGSIM_supporting_data_2.csv",
#     "python dump_data.py NGSIM_supporting_data_3.csv",
#     "python dump_data.py NGSIM_supporting_data_4.csv",
#     "python dump_data.py NGSIM_supporting_data_5.csv",
#     "python dump_data.py NGSIM_supporting_data_6.csv",
#     "python dump_data.py NGSIM_supporting_data_7.csv",
#     "python dump_data.py NGSIM_supporting_data_8.csv",
#     "python dump_data.py NGSIM_supporting_data_9.csv",
#     "python dump_data.py NGSIM_supporting_data_10.csv",
#     "python dump_data.py NGSIM_supporting_data_11.csv",
#
# ]
#
# commands = [
#     "python dump_data.py NGSIM_supporting_data_0.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_1.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_2.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_3.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_4.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_5.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_6.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_7.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_8.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_9.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_10.csv --scene i-800",
#     "python dump_data.py NGSIM_supporting_data_11.csv --scene i-800",
#
# ]

commands = [
    "python dump_data.py NGSIM_supporting_data_0.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_1.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_2.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_3.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_4.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_5.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_6.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_7.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_8.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_9.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_10.csv --scene peachtree",
    "python dump_data.py NGSIM_supporting_data_11.csv --scene peachtree",

]


for cmd in commands:
    subprocess.call(cmd, shell=True)
