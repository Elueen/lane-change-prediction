import time
import csv


with open("outcome/files/test_counter.csv", "w", newline="") as file:

    writer = csv.writer(file)

    for i in range(100):
        counter = [i+1]
        time.sleep(5)
        writer.writerow(counter)
