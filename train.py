import datetime

import torch, os
from torch.nn import functional as F
from torch.optim import *
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from model import CNN
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import time

class ModelCheckPoint:
    def __init__(self, model, model_path, mode='val_acc'):
        self.best_loss = math.inf
        self.best_acc = - math.inf
        self.best_val_loss = math.inf
        self.best_val_acc = - math.inf
        self.model = model
        self.PATH = model_path
        self.mode = mode

    def submit(self, loss, acc, val_loss, val_acc):
        if loss < self.best_loss:
            self.best_loss = loss
            if self.mode == 'loss':
                torch.save(self.model.state_dict(), self.PATH)
        if acc > self.best_acc:
            self.best_acc = acc
            if self.mode == 'acc':
                torch.save(self.model.state_dict(), self.PATH)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.mode == 'val_loss':
                torch.save(self.model.state_dict(), self.PATH)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            if self.mode == 'val_acc':
                torch.save(self.model.state_dict(), self.PATH)


class Helper:
    def __init__(self, model, optimizer, train_data, test_data, valid_ratio, batch_size, shuffle, device):
        self.model = model
        self.optim = optimizer
        self.device = device
        self.history = []
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        num_test = len(test_data)
        indices = list(range(num_test))
        split = int(np.floor(valid_ratio * num_test))

        if shuffle:
            np.random.shuffle(indices)

        test_idx, valid_idx = indices, indices[:split]
        test_sampler = SubsetRandomSampler(test_idx)

        valid_sampler = SubsetRandomSampler(valid_idx)

        self.test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)
        self.val_loader = DataLoader(test_data, batch_size=batch_size, sampler=valid_sampler)

    def fit(self, epochs, callback=None, verbose=1, mode='acc', plot=True):
        for epoch in range(epochs):
            loss, acc = self.train_epoch(epoch, self.model, self.train_loader, self.optim, self.device, verbose)
            val_loss, val_acc = self.test(self.model, self.val_loader, self.device)
            if callback is not None:
                callback.submit(loss, acc, val_loss, val_acc)
                self.history.append({'loss': loss, 'acc': acc, 'val_loss': val_loss,'val_acc': val_acc})

        if plot:
            pd.DataFrame(self.history).plot(subplots=True)

        return self.history

    def evaluate(self):
        test_loss, test_acc = self.test(self.model, self.test_loader, self.device)
        return test_loss, test_acc

    @staticmethod
    def train_epoch(epoch, model, data_loader, optimizer, device, verbose):
        model.train()
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % 100 == 0 and verbose == 1:
                print(f'Train Epoch: {epoch} Avg Loss: {total_loss/((batch_idx+1)*data_loader.batch_size)} Avg Acc: '
                      f'{100. * correct / ((batch_idx+1)*data_loader.batch_size)}')

        return total_loss/len(data_loader.dataset), correct / len(data_loader.dataset)

    @staticmethod
    def test(model, data_loader, device):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                test_loss += F.nll_loss(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(data_loader.dataset)

        print(f'\nTest set: Average loss: {test_loss}, Accuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset)}%)\n')

        return test_loss, correct / len(data_loader.dataset)


def time():
    now = datetime.datetime.now()
    name = f'{now.date()}_{now.hour}-{now.minute}'
    return name


if __name__ == '__main__':
    # a = train_data[0][0].permute(1, 2, 0)
    # print(a.size(), train_data[0][1])
    # plt.matshow(a.numpy().squeeze(-1), cmap='gray')
    # plt.show()

    PATH = '/home/aaditya/PycharmProjects/PANIIT'
    TRAIN = os.path.join(PATH, 'data', 'custom_bw', 'train')
    TEST = os.path.join(PATH, 'data', 'custom_bw', 'test')

    train_data = ImageFolder(TRAIN, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((320,320), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([127.5], [255])
    ]))

    test_data = ImageFolder(TEST, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((320,320), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([127.5], [255])
    ]))

    device = torch.device('cuda')

    model = CNN()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-2)

    helper = Helper(model, optimizer, train_data, test_data, 0.5, 32, True, device)

    NAME = 'basic_vgg_sep_cnn_spp_scratch'
    ckpt = ModelCheckPoint(model, f'{PATH}/weights/{NAME}_{time()}')

