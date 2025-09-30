import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

data = pd.read_csv('merged_data/data_214v2_merged.csv')
data.rename(columns = {'?"Time stamp"': 'TimeStamp'}, inplace = True)
mold_change = pd.read_csv('Exploration/214_mold_change.csv', )
mold_time_round = pd.to_datetime(mold_change['Time Stamp'], format='%Y/%m/%d %H:%M').dt.round('15min')
mold_change['Time Stamp'] = mold_time_round

data_ts = pd.to_datetime(data['TimeStamp'], format='%Y/%m/%d %H:%M')
data['TimeStamp'] = data_ts

small_mold_array = []
large_mold_array = []

for i in range(int(len(mold_change)/2)):
    if i == 0:
        start = mold_change.loc[i, 'Time Stamp']
        end = mold_change.loc[i+1, 'Time Stamp']
    else:
        start = mold_change.loc[2*i, 'Time Stamp']
        end = mold_change.loc[2*i+1, 'Time Stamp']

    start_idx = data.loc[data['TimeStamp'] == start, 'TimeStamp'].index[0]
    end_idx = data.loc[data['TimeStamp'] == end, 'TimeStamp'].index[0]
    # print(start_idx in data.index)
    # print(end_idx in data.index)
    small_mold_array.extend(np.array(data.loc[start_idx: end_idx, '214 motor   (kW)']))

data_slice = data[21792:]
for i in range(len(data_slice)):
    if data_slice.loc[i+21792, '214 motor   (kW)'] > 67:
        large_mold_array.append(float(data_slice.loc[i+21792, '214 motor   (kW)']))

small_mold_array = [x for x in small_mold_array if x != 0]

figure(1)
counts1, bins1, _ = plt.hist(small_mold_array, align ='mid', rwidth=0.8, bins = 20)
plt.xticks(bins1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

figure(2)
_, bins2, _ = plt.hist(large_mold_array, align = 'mid', rwidth=0.8, bins = 20)
plt.xticks(bins2)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()