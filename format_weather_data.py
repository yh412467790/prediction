import os
import csv
import time

data_set = set()


def format_weather(origin_file, target_file):
    for line in origin_file:
        writer=csv.writer(target_file)
        row = line.strip().split()
        time_array = time.strptime(row[0], "%Y%m%d%H%M")
        key = time.mktime(time_array)
        if key in data_set:
            continue
        data_set.add(key)
        row[0] = time.strftime("%Y-%m-%d %H:%M", time_array)
        writer.writerow(row)
        print(row)


path = 'data/weather_data/'
files = os.listdir(path)
for file_ in files:
    if not os.path.isdir(path + file_):
        f_name = str(file_)
        #print(f_name)
        origin_file = open(path+f_name, 'r', encoding='gbk')
        target_file = open('target_data/weather.csv', 'a+', newline="")
        format_weather(origin_file, target_file)
        origin_file.close()
        target_file.close()


