import torch
from pretrain_dataset import DatasetWrapper
from model.tvts_bert import TVTSBERT
from pretrain import TVTSBERTTrainer
import numpy as np
import random


# 每次预训练之前，改一下pretraining.py里面writter的地址
# 还要改一下模型保存地址checkpoints
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(123)

dataset_path = '../data/processed_pretrain_pems_72.csv'
pretrain_path = '../checkpoints/pretrain/'
valid_rate = 0.2
max_length = 72
word_len = 6
num_features = 1
epochs = 50
batch_size = 64
hidden_size = 256
layers = 3
attn_heads = 8
learning_rate = 1e-4
warmup_epochs = 10
decay_gamma = 0.99
dropout = 0.1
gradient_clipping = 5.0



print("Loading training and validation data sets...")
dataset = DatasetWrapper(batch_size=batch_size,
                         valid_ratio=valid_rate,
                         data_path=dataset_path,
                         num_features=num_features,
                         max_length=max_length,
                         word_len=word_len)

# training set split
train_loader, valid_loader = dataset.get_data_loaders()

print("Initialing TVTS-BERT...")
tvtsbert = TVTSBERT(num_features=num_features,
                    hidden=hidden_size,
                    n_layers=layers,
                    attn_heads=attn_heads,
                    dropout=dropout)

trainer = TVTSBERTTrainer(tvtsbert, num_features=num_features,
                          train_dataloader=train_loader,
                          valid_dataloader=valid_loader,
                          lr=learning_rate,
                          warmup_epochs=warmup_epochs,
                          decay_gamma=decay_gamma,
                          gradient_clipping_value=gradient_clipping)

print("Pretraining TVTS-BERT...")
mini_loss = np.Inf
for epoch in range(epochs):
    train_loss, valid_loss = trainer.train(epoch)
    if mini_loss > valid_loss:
        mini_loss = valid_loss
        trainer.save(epoch, pretrain_path)