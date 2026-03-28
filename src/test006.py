import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="name of csv file")
parser.add_argument("--threshold", help="number which above threshold", default=2000)
args = parser.parse_args()

df = pd.read_csv(args.file_name)

result_list = df[df.iloc[:, 1] > args.threshold][df.columns[0]].tolist()

with open('test_vel.json', 'w') as json_file:
    json.dump(result_list, json_file)