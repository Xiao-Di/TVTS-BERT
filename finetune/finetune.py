import numpy as np
from tqdm import tqdm
import utils
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model.tvts_bert import TVTSBERT
from classification_model import TVTSBERTClassification

def avg_acc(matrix):
    correct = np.diag(matrix)
    all = matrix.sum(axis=0)
    acc = correct / all
    avg_acc = np.average(acc)
    return avg_acc


class TVTSBERTFineTuner:
    def __init__(self, tvtsbert: TVTSBERT, num_classes: int,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 lr: float=1e-4, with_cuda: bool=True,
                 cuda_devices=None, log_freq: int=10):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.tvtsbert = tvtsbert
        self.model = TVTSBERTClassification(tvtsbert, num_classes).to(self.device)
        self.num_classes = num_classes

        # 多gpu并行操作
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUs for model pretraining" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # 每次finetune之前改一下writer的地址
        self.writer = SummaryWriter('../runs/2021.3.30-finetune72')
        self.log_freq = log_freq


    def train(self, epoch):

        # 进度条
        data_iter = tqdm(enumerate(self.train_dataloader),
                         desc="EP_%s:%d" % ("train", epoch),
                         total=len(self.train_dataloader),
                         bar_format="{l_bar}{r_bar}")

        train_loss = 0.0
        counter = 0
        total_correct = 0
        total_element = 0
        matrix = np.zeros([self.num_classes, self.num_classes])

        # for name, param in self.model.named_parameters():
        #     print(name)
        #     print(param.size())

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            # 将数据输入分类模型得到分类结果
            classification = self.model(data['bert_input'].float(),
                                        data['bert_mask'].long())

            loss = self.criterion(classification, data['class_label'].squeeze().long())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            train_loss += loss.item()

            # 打印输出
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": train_loss / (i+1),
                "loss": loss.item()
            }
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))


            classification_result = classification.argmax(dim=-1)
            classification_target = data['class_label'].squeeze()
            correct = classification_result.eq(classification_target).sum().item()

            total_correct += correct
            total_element += data['class_label'].nelement()
            for row, col in zip(classification_result, classification_target):
                matrix[row, col] += 1

            counter += 1

        train_loss /= counter
        train_overall_acc = total_correct / total_element * 100
        self.writer.add_scalar('train_loss', train_loss, global_step=epoch)
        self.writer.add_scalar('train_overall_acc', train_overall_acc, global_step=epoch)

        valid_loss, valid_overall_acc = self._validate()
        self.writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
        self.writer.add_scalar('valid_overall_acc', valid_overall_acc, global_step=epoch)

        print("EP%d, train_loss=%.2f, train_overall_acc=%.2f, valid_loss=%.2f, valid_overall_acc=%.2f"
              % (epoch, train_loss, train_overall_acc, valid_loss, valid_overall_acc))

        return train_loss, train_overall_acc, valid_loss, valid_overall_acc


    def _validate(self):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            total_correct = 0
            total_element = 0
            matrix = np.zeros([self.num_classes, self.num_classes])
            for data in self.valid_dataloader:
                data = {key :value.to(self.device) for key, value in data.items()}

                classification = self.model(data['bert_input'].float(),
                                            data['bert_mask'].long())

                loss = self.criterion(classification, data['class_label'].squeeze().long())
                valid_loss += loss.item()

                classification_result = classification.argmax(dim=-1)
                classification_target = data['class_label'].squeeze()

                correct = classification_result.eq(classification_target).sum().item()
                total_correct += correct
                total_element += data['class_label'].nelement()
                for r, c in zip(classification_result, classification_target):
                    matrix[r, c] += 1

                counter += 1

            valid_loss /= counter
            valid_overall_acc = total_correct / total_element * 100

        self.model.train()

        return valid_loss, valid_overall_acc

    def test(self, data_loader):
        with torch.no_grad():

            self.model.eval()

            total_correct = 0
            total_element = 0
            matrix = np.zeros([self.num_classes, self.num_classes])
            for data in data_loader:
                data = {key :value.to(self.device) for key, value in data.items()}

                result = self.model(data['bert_input'].float(),
                                    data['bert_mask'].long())

                # loss = self.criterion(classification, data['class_label'].squeeze().long())
                # valid_loss += loss.item()

                classification_result = result.argmax(dim=-1)
                classification_target = data['class_label'].squeeze()

                correct = classification_result.eq(classification_target).sum().item()
                total_correct += correct
                total_element += data['class_label'].nelement()
                for r, c in zip(classification_result, classification_target):
                    matrix[r, c] += 1

            test_overall_acc = total_correct / total_element * 100
            test_avg_acc = avg_acc(matrix)

        self.model.train()

        return test_overall_acc, test_avg_acc


    def save(self, epoch, file_path):
        output_path = file_path + "classification_checkpoint.tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict()
            }, output_path)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, file_path):
        input_path = file_path + "classification_checkpoint.tar"

        checkpoint = torch.load(input_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        print("EP:%d Model Loaded from:" % epoch, input_path)
        return input_path


