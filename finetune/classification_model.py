import torch
import torch.nn as nn
from model.tvts_bert import TVTSBERT

class TVTSBERTClassification(nn.Module):
    """
    Downstream finetune task: weekday or weekend classification
    """

    def __init__(self, tvtsbert: TVTSBERT, num_classes):
        super().__init__()
        self.tvtsbert = tvtsbert
        self.classification = MulticlassClassification(self.tvtsbert.hidden, num_classes)

    def forward(self, x, mask): # data['bert_input'], data['bert_mask']
        x = self.tvtsbert(x, mask)
        return self.classification(x)



class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes):
        super().__init__()
        self.pooling = nn.MaxPool1d(72)  # 可以换成avgpooling对比看看效果
        # self.pooling = nn.MaxPool1d(96) # 可以换成avgpooling对比看看效果
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pooling(x).squeeze()
        x = self.linear(x)
        return x