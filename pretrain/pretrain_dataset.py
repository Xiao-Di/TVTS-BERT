from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import pandas as pd
import random

# file_path = '/Users/gengyunxin/Documents/项目/traffic_model/test/data/normalized_pretrain_data_72.csv'
class PretrainDataset(Dataset):
    def __init__(self,
                 # file_path='/Users/gengyunxin/Documents/项目/traffic_model/test/data/processed_pretrain_pems_72.csv',
                 file_path='/home/user/gyx/traffic_model/test/data/processed_pretrain_pems_72.csv',
                 num_features = 1, seq_len=72, word_len=6):
        self.seq_len = seq_len
        self.word_len = word_len
        self.dimension = num_features

        df = pd.read_csv(file_path)
        self.Data = df.values # (591060, 72)
        print("Loading data successfully!")
        self.TS_num = self.Data.shape[0]

    def __len__(self):
        return self.TS_num

    def __getitem__(self, index):
        # ts_data = self.Data[:, 0:-1] # 不含label的数组 (4880,72)
        ts_data = self.Data
        # Normalize
        # max = ts_data.max() # 400.0
        # min = ts_data.min() # 3.0
        # ts_data_normalized = (ts_data - min) / (max - min)

        ts_processed = np.expand_dims(np.array(ts_data[index]), -1) # (72,1)
        # class_label = np.array([self.Data[index][-1]], dtype=int) # label (1,)

        ts_length = ts_processed.shape[0] # 72
        num_words = int(ts_length / self.word_len)

        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1 # 其实seq_len和ts_length长度相同，都是72,全1数组。这么写是为了应对长度不等的情况 (72,)

        # 随机噪声
        ts_masking, mask = self.random_masking(ts_processed, ts_length)

        output = {"bert_input": ts_masking, # 加噪声的时间序列 (72,1)
                  "bert_target": ts_processed, # 原来未加噪声的时间序列 (72,1)
                  "bert_mask": bert_mask, # 有数据的地方是1（长度ts_length）,其他地方是0（全长seq_len） (72,)
                  "loss_mask": mask, # 只计算加噪声位置的loss (72,),加噪声的位置是1,其余位置是0
                  # "class_label": class_label # (1,)
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}

    # 加入随机噪声
    # def random_masking(self, ts, ts_length):

    #     ts_masking = ts.copy()
    #     mask = np.zeros((self.seq_len,), dtype=int)

    #     for i in range(ts_length):
    #         prob = random.random()
    #         if prob < 0.15:
    #             prob /= 0.15
    #             mask[i] = 1

    #             if prob < 0.5:
    #                 ts_masking[i, :] += np.random.uniform(low=-0.5, high=0, size=(self.dimension,))

    #             else:
    #                 ts_masking[i, :] += np.random.uniform(low=0, high=0.5, size=(self.dimension,))

    #     return ts_masking, mask

    def random_masking(self, ts, num_words):

        ts_masking = ts.copy()
        mask = np.zeros((self.seq_len,), dtype=int)

        for i in range(num_words):
            prob = random.random()
            if prob < 0.2:
                prob /= 0.2
                mask[(i * self.word_len):(i * self.word_len + self.word_len)] = 1

                if prob < 0.5:
                    ts_masking[(i * self.word_len):(i * self.word_len + self.word_len), :] \
                        += np.random.uniform(low=-0.5, high=0, size=(self.dimension,))
                else:
                    ts_masking[(i * self.word_len):(i * self.word_len + self.word_len), :] \
                        += np.random.uniform(low=0, high=0.5, size=(self.dimension,))

        return ts_masking, mask




np.random.seed(0)



class DatasetWrapper(object):
    """划分训练集和验证集"""
    def __init__(self, batch_size, valid_ratio, data_path, num_features, max_length, word_len):
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.data_path = data_path
        self.num_features = num_features
        self.max_length = max_length
        self.word_len = word_len

    def get_data_loaders(self):
        dataset = PretrainDataset(self.data_path, self.num_features, self.max_length, self.word_len)
        train_loader, valid_loader = self.get_train_valid_data_loaders(dataset)
        return train_loader, valid_loader

    def get_train_valid_data_loaders(self, dataset):
        num_data = len(dataset) # 4880
        indices = list(range(num_data))
        np.random.shuffle(indices)

        valid_split = int(np.floor(self.valid_ratio * num_data))
        print('training samples: %d, validation samples: %d' % (num_data-valid_split, valid_split))
        train_idx, valid_idx = indices[valid_split:], indices[:valid_split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                 drop_last=True, num_workers=0)
        valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                 drop_last=True, num_workers=0)

        return train_loader, valid_loader

