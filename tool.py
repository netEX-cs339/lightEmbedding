import random
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms


def set_device(gpu_id):
    return torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_l_a(x, train_loss, train_acc, test_loss, test_acc, path):
    plt.clf()

    plt.subplot(121)
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(x, train_acc, 'b-', label='train')
    plt.plot(x, test_acc, 'r-', label='test')
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, train_loss, 'b-', label='train')
    plt.plot(x, test_loss, 'r-', label='test')
    plt.legend(loc='upper right')
    plt.savefig("{}/loss-acc.png".format(path))


def check_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def plot_loss(x, train_loss, test_loss, path):
    plt.clf()
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, train_loss, 'b-', label='train')
    plt.plot(x, test_loss, 'r-', label='test')
    plt.legend(loc='upper right')
    plt.savefig("{}/loss.png".format(path))


def normalize(data, channel):
    for i in range(0, channel):
        length = np.linalg.norm(data[i], ord=2)
        data[i] /= length
    return data


def update_lr(optimizer, lr):
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    for ix, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return

import pickle

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_test_init(data_path='/home/data1/liuzexu/resnet/data/', train_batch_size=64, test_batch_size=100, to_train=True, to_test=True):
    train_loader = []
    test_loader = []
    if to_train:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=0)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if to_test:
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def init_test(data_path='/home/data1/liuzexu/data', train_batch_size=64):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=False, num_workers=0)

    return train_loader


def normalization(data, channel):
    for i in range(0, channel):
        _range = np.max(data[i]) - np.min(data[i])
        data[i] = (data[i] - np.min(data[i])) / _range
    return data


class AverageValueMeter(object):
    def __init__(self, gpu_id, dim=4096):
        self.count = 0
        self.sum = torch.zeros(dim).to(gpu_id)
        self.squared_sum = torch.zeros(dim).to(gpu_id)

    def add(self, x):
        cnt = x.shape[0]
        self.count += cnt
        for i in range(0, cnt):
            self.sum += x[i].view(-1)
            self.squared_sum += x[i].view(-1) * x[i].view(-1)

    def get_var(self):
        if self.count == 0:
            raise Exception("No data!")
        return (self.squared_sum / self.count - (self.sum / self.count) * (self.sum / self.count)).mean()

    def get_std(self):
        return torch.sqrt(self.get_var())

