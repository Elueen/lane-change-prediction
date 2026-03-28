import numpy as np
import pandas as pd
import csv

nested_data1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
nested_data2 = np.array([[2, 2, 3], [4, 5, 6], [7, 8, 9]])
nested_data3 = np.array([[3, 2, 3], [4, 5, 6], [7, 8, 9]])

nd = [nested_data1, nested_data2, nested_data3]


test_data = []
for i in range(len(nd)):
    test_data.append(pd.DataFrame(np.array(nd[i])))


with open('output01.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(test_data)
