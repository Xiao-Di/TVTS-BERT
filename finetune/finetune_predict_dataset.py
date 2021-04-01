from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random

# file = "Users/gengyunxin/Documents/项目/traffic_model/gitee/data/finetune_predict_pems_144.csv"
class FinetunePredictDataset(Dataset):

    def __init__(self, file_path, seq_len, prediction_len,
                 num_features=1, word_len=6):
        # seq_len = 72, prediction_len = 12
        self.seq_len = seq_len
        self.prediction_len = prediction_len
        self.word_len = word_len
        self.dimension = num_features
        # self.mode = mode

        df = pd.read_csv(file_path)
        self.Data = df.values
        print("Loading data from " + file_path + " successfully!")
        self.TS_num = self.Data.shape[0]

    def __len__(self):
        return self.TS_num

    def __getitem__(self, index):
        """
        mode='short' --> seq_len=72+12
        mode='medium' --> seq_len=72+36
        mode='long' --> seq_len=72+72
        """

        # Solution 1: 输入输出长度84，其中后12点用0替代
        # ts_data = self.Data
        # ### 要做归一化！！而且训练验证测试集上都要用训练集的最大最小值
        # max = 124
        # min = 0
        # ts_data_normalized = (ts_data - min) / (max - min)
        # ts_array = np.array(ts_data_normalized[index])
        # ts_masked = np.append(ts_array[:self.seq_len], np.array([0]*self.prediction_len))
        # bert_input = np.expand_dims(ts_masked, -1) # (84,1)
        #
        # # ts_masking = self.masking(ts_processed[:84])
        # # bert_input = np.append(ts_processed[:72], np.array([100]*12))
        # ts_length = len(bert_input)
        # bert_mask = np.ones((ts_length,), dtype=int)
        # # bert_mask[:ts_length] = 1
        #
        # # bert_mask = np.expand_dims(bert_mask, -1)
        # loss_mask = np.array([0]*self.seq_len + [1]*self.prediction_len)
        # # bert_target = ts_processed[:ts_length]
        # bert_target = np.expand_dims(ts_array[:ts_length], -1)

        # 随机噪声
        # ts_masking, mask = self.random_masking(ts_processed, ts_length)


        # Solution 2: 输入(bert_input)72点，输出(bert_target)后12点
        ts_data = self.Data
        ### 要做归一化！！而且训练验证测试集上都要用训练集的最大最小值
        max = 124
        min = 0
        ts_data_normalized = (ts_data - min) / (max - min)
        ts_array = np.array(ts_data_normalized[index])
        ts_masked = ts_array[:self.seq_len]
        bert_input = np.expand_dims(ts_masked, -1)  # (72,1)
        bert_input84 = np.expand_dims(ts_array[:self.seq_len+self.prediction_len], -1)

        # ts_masking = self.masking(ts_processed[:84])
        # bert_input = np.append(ts_processed[:72], np.array([100]*12))
        ts_length = len(bert_input)
        bert_mask = np.ones((self.seq_len,), dtype=int)
        # bert_mask[:ts_length] = 1

        # bert_mask = np.expand_dims(bert_mask, -1)
        loss_mask = np.array([1] * self.prediction_len)
        # bert_target = ts_processed[:ts_length]
        bert_target = np.expand_dims(ts_array[self.seq_len:self.seq_len+self.prediction_len], -1)

        output = {
                  "bert_input": bert_input, # 时间序列 (72,1)
                  "bert_input84": bert_input84, # (84,1)
                  "bert_mask": bert_mask, # 有数据的地方是1,其他地方是0（全长seq_len） (72,)
                  "loss_mask": loss_mask, # 只计算要预测位置的loss (12,),预测的位置是1,其余位置是0
                  "bert_target": bert_target # (12,1)
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}


    # 加入随机噪声
    def masking(self, ts):

        ts_masking = ts.copy()
        mask = np.zeros((self.seq_len,), dtype=int)

        for i in range(12):
            prob = random.random()
            # if prob < 0.15:
            #     prob /= 0.15
            #     mask[i] = 1

            if prob < 0.5:
                ts_masking[72+i, :] += np.random.uniform(low=-0.5, high=0, size=(self.dimension,))

            else:
                ts_masking[72+i, :] += np.random.uniform(low=0, high=0.5, size=(self.dimension,))

        return ts_masking