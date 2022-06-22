import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image

from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        from torchvision import models
        self.model = models.resnet50(num_classes=60)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('densenet' * 100)

        self.domain_classifier = nn.Sequential(
            nn.Linear(864, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

        self.feature = nn.ModuleList(list((self.model.children()))[:-1])
        self.classifier = list(self.model.children())[-1]

    def forward(self, x, alpha=5):
        for m in self.feature:
            x = m(x)
        x = x.reshape(-1, 864)
        reversed_feature = ReverseLayerF.apply(x, alpha)
        class_out = self.classifier(x)
        domain_out = self.domain_classifier(reversed_feature)
        return class_out, domain_out

    def load_model(self):
        if os.path.exists('model.pth'):
            start_state = torch.load('model.pth', map_location=self.device)
            self.model.load_state_dict(start_state)
            print('using loaded model')
            print('-' * 100)

        if os.path.exists('domain_classifier.pth'):
            start_state = torch.load('domain_classifier.pth', map_location=self.device)
            self.domain_classifier.load_state_dict(start_state)
            print('using loaded domain_classifier')
            print('-' * 100)

    def save_model(self):
        result = self.model.state_dict()
        torch.save(result, 'model.pth')
        result = self.domain_classifier.state_dict()
        torch.save(result, 'domain_classifier.pth')


class Solver():
    def __init__(self,
                 batch_size=64,
                 lr=1e-3,
                 weight_decay=1e-4,
                 train_image_path='./public_dg_0416/train/',
                 valid_image_path='./public_dg_0416/train/',
                 label2id_path='./dg_label_id_mapping.json',
                 test_image_path='./public_dg_0416/public_test_flat/'
                 ):
        self.result = {}
        from data.data import get_loader, get_test_loader
        self.train_loader = get_loader(batch_size=batch_size,
                                       valid_category=None,
                                       train_image_path=train_image_path,
                                       valid_image_path=valid_image_path,
                                       label2id_path=label2id_path)
        self.test_loader_predict, _ = get_test_loader(batch_size=batch_size,
                                                      transforms=None,
                                                      label2id_path=label2id_path,
                                                      test_image_path=test_image_path)
        self.test_loader_student, self.label2id = get_test_loader(batch_size=batch_size,
                                                                  transforms='train',
                                                                  label2id_path=label2id_path,
                                                                  test_image_path=test_image_path)
        # self.train_loader = MixLoader([self.train_loader, self.test_loader_student])
        # del self.test_loader_student
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model().to(self.device)

        if os.path.exists('model.pth'):
            self.model.load_model()

        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    def save_result(self, epoch=None):
        from data.data import write_result
        result = {}
        for name, pre in list(self.result.items()):
            _, y = torch.max(pre, dim=1)
            result[name] = y.item()

        if epoch is not None:
            write_result(result, path='prediction' + str(epoch) + '.json')
        else:
            write_result(result)

        return result

    def predict(self):
        with torch.no_grad():
            print('teacher are giving his predictions!')
            self.model.eval()
            for x, names in tqdm(self.test_loader_predict):
                x = x.to(self.device)
                x, _ = self.model(x)
                for i, name in enumerate(list(names)):
                    self.result[name] = x[i, :].unsqueeze(0)  # 1, D

            print('teacher have given his predictions!')
            print('-' * 100)

    def train(self,
              total_epoch=3,
              label_smoothing=0.2,
              fp16=True,
              lam=0.2,
              ):
        from CutMix import cutmix
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        prev_loss = 999
        train_loss = 99
        criterion = nn.CrossEntropyLoss().to(self.device)
        for epoch in range(1, total_epoch + 1):
            #             # first, predict
            #             self.predict()
            #             self.save_result()

            self.model.train()
            prev_loss = train_loss
            train_loss = 0
            domain_loss = 0
            train_acc = 0
            step = 0
            pbar = tqdm(self.train_loader)
            for x, y, domain_y in pbar:
                # x, y = cutmix(x, y)
                x = x.to(self.device)
                y = y.to(self.device)
                domain_y = domain_y.to(self.device)

                if fp16:
                    with autocast():
                        x, domain_x = self.model(x)  # N, 60
                        _, pre = torch.max(x, dim=1)
                        domain_loss = criterion(domain_x, domain_y)
                        nature_loss = criterion(x, y)
                        loss = lam * domain_loss + nature_loss
                else:
                    raise NotImplementedError
                    x = self.model(x)  # N, 60
                    _, pre = torch.max(x, dim=1)
                    loss = criterion(x, y)

                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += nature_loss.item()
                domain_loss += domain_loss.item()
                self.optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()

                step += 1
                if step % 10 == 0:
                    pbar.set_postfix_str(f'nature loss = {train_loss / step}, '
                                         f'domain loss = {domain_loss / step}'
                                         f'acc = {train_acc / step}')

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            domain_loss /= len(self.train_loader)

            print(f'epoch {epoch}, nature loss = {train_loss}, '
                  f'domain loss = {domain_loss}, '
                  f'acc = {train_acc}')

            self.model.save_model()

    @torch.no_grad()
    def TTA(self, total_epoch=10, aug_weight=0.5):
        self.predict()
        print('now we are doing TTA')
        for epoch in range(1, total_epoch + 1):
            self.model.eval()
            for x, names in tqdm(self.test_loader_student):
                x = x.to(self.device)
                x, _ = self.model(x)
                for i, name in enumerate(list(names)):
                    self.result[name] += x[i, :].unsqueeze(0) * aug_weight  # 1, D

        print('TTA finished')
        self.save_result()
        print('-' * 100)


if __name__ == '__main__':
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument('-b', '--batch_size', default=2)
    paser.add_argument('-t', '--total_epoch', default=10)
    paser.add_argument('-l', '--lr', default=1e-4)
    paser.add_argument('--lam', default=1, type=float)
    args = paser.parse_args()
    batch_size = int(args.batch_size)
    total_epoch = int(args.total_epoch)
    lr = float(args.lr)
    lam = args.lam
    x = Solver(batch_size=batch_size, lr=lr)
    x.train(total_epoch=total_epoch, lam=lam)
