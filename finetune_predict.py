import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model.tvts_bert import TVTSBERT
from ft_prediction_model import TVTSBERTFTPrediction

class TVTSBERTFTPredictor:
    def __init__(self, tvtsbert: TVTSBERT, num_features, seq_len, prediction_len,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 lr: float=1e-4, with_cuda: bool=True,
                 cuda_devices=None, log_freq: int=10):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.tvtsbert = tvtsbert
        self.model = TVTSBERTFTPrediction(tvtsbert, num_features, seq_len, prediction_len).to(self.device)
        # self.num_classes = num_classes
        self.seq_len = seq_len
        self.prediction_len = prediction_len

        # 多gpu并行操作
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUs for model pretraining" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')

        # 每次finetune之前改一下writer的地址
        self.writer = SummaryWriter('../runs/2021.3.31-finetune_predict_72')
        self.log_freq = log_freq


    def train(self, epoch):

        # 进度条
        data_iter = tqdm(enumerate(self.train_dataloader),
                         desc="EP_%s:%d" % ("train", epoch),
                         total=len(self.train_dataloader),
                         bar_format="{l_bar}{r_bar}")

        train_loss = 0.0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            # print('shape of bert_input:', data['bert_input'].shape)
            # print('shape of bert_mask:', data['bert_mask'].shape)
            # print('shape of bert_target:', data['bert_target'].shape)
            # print('shape of loss_mask:', data['loss_mask'].shape)

            # 计算后一段时间位置处的预测值，并与真实值算loss:MSE
            finetune_prediction = self.model(data['bert_input'].float(),
                                             data['bert_mask'].long()) # (12,1)

            loss = self.criterion(finetune_prediction, data['bert_target'].float()) # nn.MSELoss(reduction='none'), (84,1)
            mask = data['loss_mask'].unsqueeze(-1) # (12,) -> (12,1)
            loss = (loss * mask.float()).sum() / mask.sum() # 对后12位需要预测对位置(全部序列)求loss的平均

            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clipping) # 防止梯度爆炸
            self.optim.step()

            train_loss += loss.item() # 取scalar，叠加

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": train_loss / (i+1), # epoch的平均loss
                "loss": loss.item() # iter的loss
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        train_loss = train_loss / len(data_iter)
        self.writer.add_scalar('train_loss', train_loss, global_step=epoch)

        valid_loss = self._validate()
        self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch)

        # warmup
        # if epoch >= self.warmup_epochs:
            # self.optim_schedule.step()
        # self.writer.add_scalar('cosine_lr_decay', self.optim_schedule.get_lr()[0], global_step=epoch) # lr第一维

        print("EP%d, train_loss=%.5f, validation_loss=%.5f" % (epoch, train_loss, valid_loss))

        return train_loss, valid_loss


    def _validate(self):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for data in self.valid_dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}

                finetune_prediction = self.model(data['bert_input'].float(),
                                                 data['bert_mask'].long())

                # print(len(finetune_prediction[0])) # 84

                loss = self.criterion(finetune_prediction, data['bert_target'].float())
                mask = data['loss_mask'].unsqueeze(-1)
                loss = (loss * mask.float()).sum() / mask.sum()

                valid_loss += loss.item()
                counter += 1

            valid_loss /= counter

        self.model.train()
        return valid_loss


    def test(self, data_loader):
        """
            取test_dataloader(1000条样本)的第i个batch中的第j个样本画图对比prediction_result和target
        """
        # if i<0 or i>len(data_loader):
        #     print("Index out of range of test dataloader!")
        # else:
        max = 124
        min = 0
        # epsilon = 1 # 避免除数为0造成inf

        with torch.no_grad():
            self.model.eval()

            prediction_result_list = []
            # prediction_target_list = []
            input84_list = []

            print('num of data in test dataloader: ', len(data_loader))
            counter = 0
            test_error = 0.0
            overall_error = 0.0

            for data in data_loader:
                data = {key: value.to(self.device) for key, value in data.items()}
                prediction_result = self.model(data['bert_input'].float(),
                                               data['bert_mask'].long()) # 12
                prediction_target = data['bert_target'].float() # 12
                input84 = data['bert_input84'].float() # 84
                # 反归一化

                prediction_result_inverse = ((max-min)*prediction_result + min).squeeze()
                prediction_target_inverse = ((max-min)*prediction_target + min).squeeze()
                input84_inverse = ((max-min)*input84 + min).squeeze()

                prediction_result_list.append(prediction_result_inverse)
                # prediction_target_list.append(prediction_target_inverse)
                input84_list.append(input84_inverse)

                epsilon = torch.ones(prediction_result_inverse.size()).to(self.device) # 避免除数为0造成inf
                # 计算error [32, 12, 1]
                error = torch.abs(prediction_target_inverse - prediction_result_inverse) / (prediction_target_inverse + epsilon)
                # error = error.squeeze() # [32,12] 最后一个是[7,12]
                # 有的位置出现了inf，替换为0
                error = torch.where(torch.isinf(error), torch.full_like(error,0.0), error)

                num_rows = 0
                for row in range(error.shape[0]): # 32
                    position = torch.ones(error[row].size()) # [12]
                    avg_error = torch.sum(error[row]) / torch.sum(position)
                    # test_error += error.item()
                    # print('error for 12 points: ', error[row])
                    print('Average error for 12 points:', avg_error)
                    num_rows += 1
                    test_error += avg_error
                test_error = test_error / num_rows
                print("-" * 50)
                print('Average error for a batch:', test_error)
                print("-" * 50)

                overall_error += test_error
                counter += 1

            overall_test_error = overall_error / counter
            print('-*-' * 10)
            print("Overall test error: ", overall_test_error)

        self.model.train()
        # return prediction_result_list, prediction_target_list
        return prediction_result_list, input84_list, test_error


    def save(self, epoch, file_path):
        output_path = file_path + "predict_checkpoint.tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict()
            }, output_path)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, file_path):
        input_path = file_path + "predict_checkpoint.tar"

        checkpoint = torch.load(input_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        print("EP:%d Model Loaded from:" % epoch, input_path)
        return input_path


