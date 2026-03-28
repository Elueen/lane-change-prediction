import pandas as pd
import argparse


def read_csv(file_path, num_lines):

    df = pd.read_csv(file_path, nrows=num_lines)

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read the first N lines of a CSV file.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file")
    parser.add_argument("--num_lines", type=int, default=1000, help="Number of lines to read (default: 1000)")
    args = parser.parse_args()

    displayed_data = read_csv(args.file_path, args.num_lines)

    print(displayed_data)
