import torch.nn as nn
from model.tvts_bert import TVTSBERT

class TVTSBERTPrediction(nn.Module):
    """
    Prediction task:predict contaminated data;
    loss: MSE
    """

    def __init__(self, tvtsbert: TVTSBERT, num_features):
        super().__init__()
        self.tvtsbert = tvtsbert
        self.mask_prediction = MaskedTimeSeriesModel(self.tvtsbert.hidden, num_features)

    def forward(self, x, mask):
        # print('x in:', x.shape)
        # print('mask in:', mask.shape)

        x = self.tvtsbert(x, mask)
        # print('x out tvtsbert:', x.shape)
        return self.mask_prediction(x)


class MaskedTimeSeriesModel(nn.Module):

    def __init__(self, hidden, num_features):
        super().__init__()
        self.linear = nn.Linear(hidden, num_features) # in_features输入二维张量大小, out_features输出二维张量大小

    def forward(self, x):
        return self.linear(x)

