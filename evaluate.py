PROJECT_ROOT = "D:/grade3/FallSemester/计算机网络/embedding"
import os
import sys

sys.path.append(PROJECT_ROOT)
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import numpy as np

import torch
import torch.utils.data as data
from torch.autograd import Variable

from VAE import VAE
from dataset import BinaryDataset


GPU_ID = 0
BATCH_SIZE = 1
LATENT_DIM = 8

DATA_ROOT = "D:/grade3/FallSemester/计算机网络/embedding/Data/bin/sample10000"
MODEL_PATH = "D:/grade3/FallSemester/计算机网络/embedding/save/models/model_100.pkl"


def main():
    dim = np.load(os.path.join(DATA_ROOT, "test", "dim.npy"))
    pretrained_model = VAE(input_dim = dim, latent_dim = LATENT_DIM).to(GPU_ID)
    pretrained_model.load_state_dict(torch.load(MODEL_PATH))

    test_dataset = BinaryDataset(DATA_ROOT, phase = 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = True)

    pretrained_model.eval()
    with torch.no_grad():
        total_differ = 0
        cnt = 0
        for record in tqdm(test_loader, desc = "Analyzing test set"):
            cnt = cnt + 1
            record = Variable(record).to(GPU_ID)

            recon_x, mu, logvar = pretrained_model.forward(record)

            recon_x = recon_x.cpu().numpy()
            recon_x = np.squeeze(recon_x)
            for i in range(0, recon_x.shape[0]):
                if recon_x[i] - 0.5 >= 0:
                    recon_x[i] = 1
                else:
                    recon_x[i] = 0
            recon_x = recon_x.astype(bool)
            x = record.cpu().numpy()
            x = np.squeeze(x).astype(bool)

            differ = sum(recon_x != x)
            total_differ = total_differ + differ
            avg_error = total_differ / (cnt*dim)

        print("Average error is :{}".format(avg_error))


if __name__ == '__main__':
    main()
