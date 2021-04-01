import torch.nn as nn
from model.tvts_bert import TVTSBERT

class TVTSBERTFTPrediction(nn.Module):
    """
    Prediction task:predict contaminated data;
    loss: MSE
    """

    def __init__(self, tvtsbert: TVTSBERT, num_features, seq_len, prediction_len):
        super().__init__()
        self.tvtsbert = tvtsbert
        self.prediction = PredictionModel(self.tvtsbert.hidden, num_features,
                                          seq_len, prediction_len)

    def forward(self, x, mask):
        # print('x in:', x.shape)
        # print('mask in:', mask.shape)

        x = self.tvtsbert(x, mask)
        # print('x out tvtsbert:', x.shape)
        return self.prediction(x)


# class PredictionModel(nn.Module):
#
#     def __init__(self, hidden, num_features, seq_len, prediction_len):
#         super().__init__()
#         self.linear1 = nn.Linear(hidden, num_features) # in_features输入二维张量大小, out_features输出二维张量大小
#         # self.linear2 = nn.Linear(seq_len, prediction_len) # 输入数据长度、预测数据长度
#
#     def forward(self, x):
#         x = self.linear1(x)
#         # print('size after linear1: ', x.shape)
#         return x


class PredictionModel(nn.Module):

    def __init__(self, hidden, num_features, seq_len, prediction_len):
        super().__init__()
        self.linear1 = nn.Linear(hidden, num_features) # in_features输入二维张量大小, out_features输出二维张量大小
        self.linear2 = nn.Linear(seq_len, prediction_len) # 输入数据长度、预测数据长度

    def forward(self, x):
        x = self.linear1(x).squeeze() # (32, 72)
        # print('size after linear1 and squeeze: ', x.shape)
        return self.linear2(x).unsqueeze(-1)

