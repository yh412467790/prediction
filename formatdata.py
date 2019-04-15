import csv
import time
origin_file = open('data/p', 'r', encoding='gbk')
target_file = open('target_data/p.csv', 'w', newline="")
writer = csv.writer(target_file)

for line in origin_file:

    row = line.strip().split()
    time_array = time.strptime(row[0]+row[1], "%Y-%m-%d%H:%M")
    row[0] = time.strftime("%Y-%m-%d %H:%M", time_array)

    if float(row[2]) < 0:
        row[2] = 0

    writer.writerow(row)

origin_file.close()