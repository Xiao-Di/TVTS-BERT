import numpy as np
import pandas as pd

def read_data(file_path, label):
    # 只读取csv中第一列车流量数据，即767838路段 (num_features=1)
    df = pd.read_csv(file_path, usecols=[0], names=None).values.tolist()
    raw_data = []
    for data in df:
        raw_data.append(data[0])
    data_per_hour = np.array(raw_data, dtype=float).reshape(-1,12)
    num_rows = data_per_hour.shape[0]
    if label == 1:
        labels = np.ones((1, num_rows))
    else:
        labels = np.zeros((1, num_rows))
    data_with_label = np.insert(data_per_hour, 12, values=labels, axis=1)

    return data_with_label

workday_file = '/Users/gengyunxin/Documents/项目/traffic_model/data/California/combine_E_workday_n0.csv'
weekend_file = '/Users/gengyunxin/Documents/项目/traffic_model/data/California/combine_E_weekend_n0.csv'
# weekday->1 ; weekend->0
workday_with_label = read_data(workday_file, label=1) # (1032, 13)
weekend_with_label = read_data(weekend_file, label=0) # (432, 13)

all_with_label = np.append(workday_with_label, weekend_with_label, axis=0) # (1464, 13)

df = pd.DataFrame(all_with_label)
df.to_csv('/Users/gengyunxin/Documents/项目/traffic_model/data/California/processed_all_data.csv', index=0, header=0)