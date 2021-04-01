import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from model.tvts_bert import TVTSBERT
from finetune_predict import TVTSBERTFTPredictor
from finetune_predict_dataset import FinetunePredictDataset
import numpy as np
import random
from matplotlib import pyplot as plt

def fig_plot(prediction_result_list, prediction_target_list, batch, sample,
             seq_len=72, prediction_len=12):
    plt.figure()
    y1 = prediction_result_list[batch][sample]
    y2 = prediction_target_list[batch][sample]
    # x1 = len(prediction_target[72:84])
    x2 = len(y2)
    plt.plot(list(range(x2)), np.append(y2[:seq_len].cpu(), y1[seq_len:seq_len+prediction_len].cpu()),
            c='r', label='prediction result')
    plt.plot(list(range(x2)), y2.cpu(), c='b', label='target')
    plt.plot(list(range(x2)), y1.cpu(), c='g', label='prediction result all')
    plt.title('batch%d-sample%d' % (batch, sample))
    plt.legend(['prediction_result', 'prediction_target', 'prediction84'])
    plt.savefig('3.31finetune_predict_0_batch%d_sample%d.png' % (batch, sample))

def test_plot(prediction_result_list, prediction_target_list, batch, sample):
    plt.figure()
    # x1 = len(prediction_target_list[batch][sample][:72]) # 横轴72
    # x2 = len(prediction_result_list[batch][sample][:12])  # 横轴12
    x1 = 72
    x2 = 12
    for i in range(4):
        plt.plot(np.arange(i*x1, (i+1)*x1),
                 prediction_target_list[batch][sample+i][:72].cpu(),
                 c='b', label='Target')
    for j in range(18):
        plt.plot(np.arange(j*x2+72, (j+1)*x2+72),
                 prediction_result_list[batch][sample+j][72:84].cpu(),
                 c='r', label='Predict')
    plt.title('batch%d-sample%d' % (batch, sample))
    # plt.legend(['prediction_target', 'prediction_result'])
    plt.savefig('3.31finetune_predict_0_batch%d_sample%d.png' % (batch, sample))

def test2_plot(prediction_result_list, input84_list, batch, sample):
    plt.figure()
    plt.plot(input84_list[batch][sample].cpu(), c='b', label='Target')
    plt.plot(np.arange(72,84), prediction_result_list[batch][sample].cpu(), c='r', label='Predict')
    plt.title('batch%d-sample%d' % (batch, sample))
    plt.legend(['prediction_target', 'prediction_result'])
    plt.savefig('3.31finetune_predict_2linear_batch%d_sample%d.png' % (batch, sample))


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

seq_len = 72
prediction_len = 12
num_features = 1
word_len =6

epochs = 100
batch_size = 32
hidden_size = 256
layers = 3
attn_heads = 8
learning_rate = 2e-5
dropout = 0.1

train_file = file_path + 'train000unnormed.csv'
valid_file = file_path + 'valid000unnormed.csv'
test_file = file_path + 'test000unnormed.csv'
# train_file = file_path + 'ftp_normed_train_144.csv'
# valid_file = file_path + 'ftp_normed_valid_144.csv'
# test_file = file_path + 'ftp_normed_test_144.csv'

# train_file = file_path + 'ftp_train_144.csv'
# valid_file = file_path + 'ftp_valid_144.csv'
# test_file = file_path + 'ftp_test_144.csv'

train_dataset = FinetunePredictDataset(file_path=train_file,
                                num_features=num_features,
                                seq_len=seq_len,
                                prediction_len=prediction_len,
                                word_len=word_len)
valid_dataset = FinetunePredictDataset(file_path=valid_file,
                                num_features=num_features,
                                seq_len=seq_len,
                                prediction_len=prediction_len,
                                word_len=word_len)
test_dataset = FinetunePredictDataset(file_path=test_file,
                               num_features=num_features,
                               seq_len=seq_len,
                               prediction_len=prediction_len,
                               word_len=word_len)

train_dataloader = DataLoader(train_dataset, shuffle=True,
                                batch_size=batch_size, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, shuffle=False,
                                batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(test_dataset, shuffle=False,
                                batch_size=batch_size, drop_last=False)

tvtsbert = TVTSBERT(num_features=num_features,
                    hidden=hidden_size,
                    n_layers=layers,
                    attn_heads=attn_heads,
                    dropout=dropout)

tvtsbert_path = pretrain_path + "checkpoint.bert.pth"
tvtsbert.load_state_dict(torch.load(tvtsbert_path))

finetuner = TVTSBERTFTPredictor(tvtsbert, num_features=num_features,
                                seq_len=seq_len, prediction_len=prediction_len,
                                train_dataloader=train_dataloader,
                                valid_dataloader=valid_dataloader)

# Test: 重新加载finetune的模型
print("\n" * 5)
print("Testing TVTS-BERT Finetune Predictor...")
finetuner.load(finetune_path)
# test需要输入参数i:第几个epoch(< 31)
# 参数j:第几个样本(< 32)

# prediction_result_list, prediction_target_list = finetuner.test(test_dataloader)
prediction_result_list, input84_list, test_error = finetuner.test(test_dataloader)
# 画图
# fig_plot(prediction_result_list, prediction_target_list, batch=5, sample=10,
#              seq_len=seq_len, prediction_len=prediction_len)

# test_plot(prediction_result_list, prediction_target_list, batch=5, sample=0)
test2_plot(prediction_result_list, input84_list, batch=20, sample=10)

