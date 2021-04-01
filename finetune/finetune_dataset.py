from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import pandas as pd
import random

# file_path = '/Users/gengyunxin/Documents/项目/traffic_model/test/data/processed_finetune_seattle_72.csv'
class FinetuneDataset(Dataset):
    """
    和PretrainDataset的区别在于：
    不需要mask，所以少了两个输入（mask后的序列和mask的位置）；
    数据集file_path变化了；
    所以也不需要word_len这个参数了，但是这版没有删掉以防后面会用到。
    """

    def __init__(self,
                 file_path, seq_len,
                 num_features = 1, word_len=6):
        self.seq_len = seq_len
        self.word_len = word_len
        self.dimension = num_features

        df = pd.read_csv(file_path)
        self.Data = df.values
        print("Loading data from " + file_path + " successfully!")
        self.TS_num = self.Data.shape[0]

    def __len__(self):
        return self.TS_num

    def __getitem__(self, index):
        ts_data = self.Data[:, 0:-1] # 不含label的数组
        # Normalize
        # max = ts_data.max() # 400.0
        # min = ts_data.min() # 3.0
        # ts_data_normalized = (ts_data - min) / (max - min)

        ts_processed = np.expand_dims(np.array(ts_data[index]), -1) # 归一化后的数据 (72,1)
        class_label = np.array([self.Data[index][-1]], dtype=int) # label (1,)

        ts_length = ts_processed.shape[0] # 72
        num_words = int(ts_length / self.word_len)

        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1 # 其实seq_len和ts_length长度相同，都是72,全1数组。这么写是为了应对长度不等的情况 (72,)

        # 随机噪声
        # ts_masking, mask = self.random_masking(ts_processed, ts_length)

        output = {
                  # "bert_input": ts_masking, # 加噪声的时间序列 (72,1)
                  "bert_input": ts_processed, # 原来未加噪声的时间序列 (72,1)
                  "bert_mask": bert_mask, # 有数据的地方是1（长度ts_length）,其他地方是0（全长seq_len） (72,)
                  # "loss_mask": mask, # 只计算加噪声位置的loss (72,),加噪声的位置是1,其余位置是0
                  "class_label": class_label # (1,)
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}