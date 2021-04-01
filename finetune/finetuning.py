import torch
from torch.utils.data import DataLoader
from model.tvts_bert import TVTSBERT
from finetune import TVTSBERTFineTuner
from finetune_dataset import FinetuneDataset
import numpy as np
import random
from matplotlib import pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

file_path = '../data/'

pretrain_path = '../checkpoints/pretrain/' # the storage path of the pretrained model
finetune_path = '../checkpoints/finetune/' # the output directory where the finetuning checkpoints written

max_length = 72
# max_length = 96
num_features = 1

num_classes = 2 # 分类任务数
# num_classes = 3 # 分类任务数
word_len =6

epochs = 50
batch_size = 32
hidden_size = 256
layers = 3
attn_heads = 8
learning_rate = 2e-5
dropout = 0.1

train_file = file_path + 'ff_train_72.csv'
valid_file = file_path + 'ff_valid_72.csv'
test_file = file_path + 'ff_test_72.csv'
# train_file = file_path + 'ff_train_96.csv'
# valid_file = file_path + 'ff_valid_96.csv'
# test_file = file_path + 'ff_test_96.csv'

print("Lodaing data sets...")
train_dataset = FinetuneDataset(file_path=train_file,
                                num_features=num_features,
                                seq_len=max_length,
                                word_len=word_len)
valid_dataset = FinetuneDataset(file_path=valid_file,
                                num_features=num_features,
                                seq_len=max_length,
                                word_len=word_len)
test_dataset = FinetuneDataset(file_path=test_file,
                               num_features=num_features,
                               seq_len=max_length,
                               word_len=word_len)

print("Creating dataloader...")
train_dataloader = DataLoader(train_dataset, shuffle=True,
                                batch_size=batch_size, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, shuffle=False,
                                batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(test_dataset, shuffle=False,
                                batch_size=batch_size, drop_last=False)

print("Initializing TVTS-BERT...")
tvtsbert = TVTSBERT(num_features=num_features,
                    hidden=hidden_size,
                    n_layers=layers,
                    attn_heads=attn_heads,
                    dropout=dropout)

print("Loading pretrained model parameters...")
tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
tvtsbert.load_state_dict(torch.load(tvtsbert_path))

print("Creating downstream classification task FineTuner...")
finetuner = TVTSBERTFineTuner(tvtsbert, num_classes=num_classes,
                              train_dataloader=train_dataloader,
                              valid_dataloader=valid_dataloader)

print("Finetuning TVTS-BERT for Classification...")
overall_acc = 0
train_loss_list = []
train_overall_acc_list = []
valid_loss_list = []
valid_overall_acc_list = []

for epoch in range(epochs):
    train_loss, train_overall_acc, valid_loss, valid_overall_acc = finetuner.train(epoch)
    if overall_acc < valid_overall_acc:
        overall_acc = valid_overall_acc
        finetuner.save(epoch, finetune_path)

    train_loss_list.append(train_loss)
    train_overall_acc_list.append(train_overall_acc)
    valid_loss_list.append(valid_loss)
    valid_overall_acc_list.append(valid_overall_acc)

fig = plt.figure()
ax1 = plt.subplot(221)
ax1.plot(list(range(epochs)), train_loss_list)
ax1.set_title('train loss/finetune')
ax1.set_xlabel('epoch')
ax1.set_ylabel('train_loss')

ax2 = plt.subplot(222)
ax2.plot(list(range(epochs)), train_overall_acc_list)
ax2.set_title('train acc/finetune')
ax2.set_xlabel('epoch')
ax2.set_ylabel('train_acc')

ax3 = plt.subplot(223)
ax3.plot(list(range(epochs)), valid_loss_list)
ax3.set_title('valid loss/finetune')
ax3.set_xlabel('epoch')
ax3.set_ylabel('valid_loss')

ax4 = plt.subplot(224)
ax4.plot(list(range(epochs)), valid_overall_acc_list)
ax4.set_title('valid acc/finetune')
ax4.set_xlabel('epoch')
ax4.set_ylabel('valid_acc')

plt.savefig('3.30_finetune72_result.png')




# Test: 重新加载finetune的模型
print("\n" * 5)
print("Testing TVTS-BERT...")
finetuner.load(finetune_path)
test_overall_acc, test_avg_acc = finetuner.test(test_dataloader)
print("test_overall_acc = %.2f, test_avg_acc = %.3f" % (test_overall_acc, test_avg_acc))