import argparse

def remove_at_sign(line):
    if "@" in line:
        return line.split("@")[0].strip()
    return line.strip()


def process_file(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    lines = [remove_at_sign(line) for line in lines]

    with open(output_file, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    input_file = "requirements.txt"
    output_file = "re_requirements.txt"

    process_file(input_file, output_file)
