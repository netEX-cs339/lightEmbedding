PROJECT_ROOT = "D:/grade3/FallSemester/计算机网络/embedding"
import os
import sys

sys.path.append(PROJECT_ROOT)
from tqdm import tqdm
import matplotlib
import torch.optim as optim

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

from tool import update_lr
from VAE import loss_func, VAE
from dataset import BinaryDataset


GPU_ID = 0
IS_TRAIN = True
BATCH_SIZE = 32
EPOCH_NUM = 10
LEARNING_RATE_TYPE = "3"
SAVE_SET = 'train'  # test
EPOCH_NUM_CALC = 100
NET = 'VAE'
DATA_SET = 'BIN'
LATENT_DIM = 256

DATA_ROOT = "D:/grade3/FallSemester/计算机网络/embedding/Data/bin/sample2000"


class NetworkTrainer(object):
    def __init__(self, args):
        self.GPU_ID = args.gpu_id
        self.IS_TRAIN = args.is_train
        self.NET = args.net_type
        self.LEARNING_RATE_TYPE = args.learning_rate
        self.BATCH_SIZE = args.batch_size
        self.NUM_EPOCHS = args.num_epochs
        self.NUM_CALC_EPOCHS = args.num_calc_epochs
        self.SAVE_SET = args.save_set
        self.CALC_EPOCHS = torch.linspace(0, self.NUM_EPOCHS, self.NUM_CALC_EPOCHS + 1)
        self.DATA_ROOT = DATA_ROOT
        self.DATA_SET = DATA_SET
        self.LATENT_DIM = LATENT_DIM

        self.LR_LIST_3 = np.ones(args.num_epochs) * 1e-3  # arg:'3'
        self.LR_LIST_45 = np.logspace(-4, -5, args.num_epochs)  # arg:'45'
        self.LR_LIST_56 = np.logspace(-5, -6, args.num_epochs)  # arg:'56'
        self.LR_LIST_23 = np.logspace(-2, -3, args.num_epochs)  # arg:'23'
        self.LR_LIST_34 = np.logspace(-3, -4, args.num_epochs)  # arg:'34'
        self.LR_LIST_67 = np.logspace(-6, -7, args.num_epochs)  # arg:'67'
        self.LR_LIST_89 = np.logspace(-8, -9, args.num_epochs)  # arg:'89'

        self.PATH = {}
        self.PATH['fig_save_path'] = PROJECT_ROOT + '/save/figs/'
        self.PATH['fig_save_path_curve'] = self.PATH['fig_save_path'] + 'curve.png'
        self.PATH['model_save_path'] = PROJECT_ROOT + '/save/models/'
        self.PATH['model_save_path_net'] = self.PATH['model_save_path'] + 'model.pkl'
        self.PATH['data_save_path'] = PROJECT_ROOT + '/save/data/{}_{}/'.format(self.NET, self.DATA_SET)
        self.PATH['data_save_path_log'] = self.PATH['data_save_path'] + 'log.pkl'

        if not os.path.exists(self.PATH['fig_save_path']):
            os.makedirs(self.PATH['fig_save_path'])
        if not os.path.exists(self.PATH['model_save_path']):
            os.makedirs(self.PATH['model_save_path'])
        if not os.path.exists(self.PATH['data_save_path']):
            os.makedirs(self.PATH['data_save_path'])
        self.EPOCH = 1

    def prepare(self):
        torch.cuda.set_device(self.GPU_ID)
        self._prepare_model()
        self._prepare_dataset()
        self._generate_plot_dic()

    def _prepare_model(self):
        dim = np.load(os.path.join(self.DATA_ROOT, "train", "dim.npy"))
        if self.NET == 'VAE':
            self.model = VAE(input_dim = dim, latent_dim = self.LATENT_DIM).to(self.GPU_ID)

        if self.LEARNING_RATE_TYPE == '45':
            self.LEARNING_RATE = self.LR_LIST_45
        elif self.LEARNING_RATE_TYPE == '67':
            self.LEARNING_RATE = self.LR_LIST_67
        elif self.LEARNING_RATE_TYPE == '89':
            self.LEARNING_RATE = self.LR_LIST_89
        elif self.LEARNING_RATE_TYPE == '34':
            self.LEARNING_RATE = self.LR_LIST_34
        elif self.LEARNING_RATE_TYPE == '23':
            self.LEARNING_RATE = self.LR_LIST_23
        elif self.LEARNING_RATE_TYPE == '56':
            self.LEARNING_RATE = self.LR_LIST_56
        elif self.LEARNING_RATE_TYPE == '3':
            self.LEARNING_RATE = self.LR_LIST_3

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE[0], betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0)

    def _prepare_dataset(self):
        print(self.DATA_ROOT)

        # load the data
        train_dataset = BinaryDataset(self.DATA_ROOT, phase='train')
        test_dataset = BinaryDataset(self.DATA_ROOT, phase='test')

        # encapsulate them into dataloader form
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.BATCH_SIZE, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False,
                                                       drop_last=True)

    def _generate_plot_dic(self):
        self.plot_dict = {
            'train_loss': [],
            'test_loss': []
        }

    def train_epoch(self):
        self.model.train()
        print("Learning rate is", self.LEARNING_RATE[self.EPOCH - 1])
        update_lr(self.optimizer, self.LEARNING_RATE[self.EPOCH - 1])
        total_loss = 0
        cnt = 0
        for record in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            record = Variable(record).to(self.GPU_ID)
            self.optimizer.zero_grad()

            if self.EPOCH == 1 and cnt == 0:
                self.save(epoch=0)
            cnt += 1

            recon_x, mu, logvar = self.model.forward(record)
            loss = loss_func(recon_x, record, mu, logvar)
            loss.backward()
            total_loss += loss
            self.optimizer.step()

        length = len(self.train_loader) // self.BATCH_SIZE
        avg_loss = float(total_loss) / length

        self.plot_dict['train_loss'].append(avg_loss)
        print('ELBO of the network on the training records: {}'.format(avg_loss))
        return avg_loss

    def eval_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for record in tqdm(self.test_loader, desc="evaluation " + str(self.EPOCH), mininterval=1):
                record = Variable(record).to(self.GPU_ID)

                recon_x, mu, logvar = self.model.forward(record)
                loss = loss_func(recon_x, record, mu, logvar)
                total_loss += loss

        length = len(self.test_loader.dataset) // self.BATCH_SIZE
        avg_loss = float(total_loss) / length

        self.plot_dict['test_loss'].append(avg_loss)
        print('ELBO of the network on the testing records: {}'.format(avg_loss))
        return avg_loss

    def draw(self):
        print('ploting...')
        plt.figure(figsize=(8, 8))
        x = range(1, len(self.plot_dict["train_loss"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dict["train_loss"], label="train_loss")
        plt.plot(x, self.plot_dict["test_loss"], label="test_loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path_curve'], bbox_inches='tight', dpi=300)

    def save(self, epoch):
        print('saving...')
        torch.save(self.model.state_dict(), self.PATH['model_save_path'] + 'model_{}.pkl'.format(epoch))

        with open(self.PATH['data_save_path_log'], 'wb') as f:
            pickle.dump(self.plot_dict, f)

    def run(self):
        self.prepare()
        for self.EPOCH in range(1, self.NUM_EPOCHS + 1):
            self.train_epoch()
            self.eval_epoch()
            if self.EPOCH % 5 == 0:
                self.draw()
                self.save(self.EPOCH)


def main():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--gpu-id', type=int, default=GPU_ID)
    parser.add_argument('--is-train', type=bool, default=IS_TRAIN)
    parser.add_argument('--net_type', type=str, default=NET)
    parser.add_argument('--learning-rate', type=str, default=LEARNING_RATE_TYPE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-epochs', type=int, default=EPOCH_NUM)
    parser.add_argument('--save-set', type=str, default=SAVE_SET)
    parser.add_argument('--num-calc-epochs', type=int, default=EPOCH_NUM_CALC)

    args = parser.parse_args()
    print(args)

    if args.is_train:
        net = NetworkTrainer(args)
        net.run()


if __name__ == '__main__':
    main()
