
import pandas as pd
import csv
import time
import datetime
f = open('data/sp_500.csv')
df = pd.read_csv(f)
data = df.iloc[:, :].values
target_file = open('data/train.csv', 'w', newline="")
# target_file = open('check.csv', 'w', newline="")
writer = csv.writer(target_file)
times=[]
for d in data:
    row = []
    time_array = time.strptime(d[0], "%m/%d/%Y")
    row.append(str(time_array[0]))
    row.append(str(time_array[1]))
    row.append(str(time_array[2]))
    row.append(d[1])
    row.append(d[2])
    row.append(d[3])
    row.append(d[4])
    row.append(d[5])
    writer.writerow(row)
