PROJECT_ROOT = "D:/grade3/FallSemester/计算机网络/embedding"
import os
import sys
import pickle

sys.path.append(PROJECT_ROOT)
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.utils.data as data
from torch.autograd import Variable

from VAE import VAE
from dataset import BinaryDataset

from sklearn.decomposition import PCA


GPU_ID = 0
BATCH_SIZE = 1
LATENT_DIM = 8

DATA_ROOT = "D:/grade3/FallSemester/计算机网络/embedding/Data/bin/sample10000"
MODEL_PATH = "D:/grade3/FallSemester/计算机网络/embedding/save/models/model_100.pkl"
SAVE_PATH = "D:/grade3/FallSemester/计算机网络/embedding/save/figs"


def pca(input):
    machine = PCA(n_components = 2)
    machine.fit(input)
    X = machine.transform(input)
    return X


def main():
    china = []
    germany = []
    japan = []
    akamai_ghost = []
    apache = []
    router = []
    trusted = []
    untrusted = []

    with open('Data/bin/label_location.pkl', 'rb') as f:
        location_dict = pickle.load(f)
    with open('Data/bin/label_server.pkl', 'rb') as f:
        server_dict = pickle.load(f)
    with open('Data/bin/label_certificate.pkl', 'rb') as f:
        certificate_dict = pickle.load(f)


    dim = np.load(os.path.join(DATA_ROOT, "test", "dim.npy"))
    pretrained_model = VAE(input_dim = dim, latent_dim = LATENT_DIM).to(GPU_ID)
    pretrained_model.load_state_dict(torch.load(MODEL_PATH))

    train_dataset = BinaryDataset(DATA_ROOT, phase = 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = True)

    pretrained_model.eval()
    with torch.no_grad():
        cnt = 0
        for record in tqdm(train_loader, desc = "Analyzing test set"):
            record = Variable(record).to(GPU_ID)

            _, mu, _ = pretrained_model.forward(record)

            mu = mu.cpu().numpy()
            mu = np.squeeze(mu)

            if cnt in location_dict["China"]:
                china.append(mu)
            elif cnt in location_dict["Germany"]:
                germany.append(mu)
            elif cnt in location_dict["Japan"]:
                japan.append(mu)

            if cnt in server_dict["AkamaiGHost"]:
                akamai_ghost.append(mu)
            elif cnt in server_dict["Apache"]:
                apache.append(mu)
            elif cnt in server_dict["Router"]:
                router.append(mu)

            if cnt in certificate_dict[True]:
                trusted.append(mu)
            elif cnt in certificate_dict[False]:
                untrusted.append(mu)

            cnt = cnt + 1

    print("start PCA....")
    location = china + germany + japan
    location_2d = pca(location)

    china = location_2d[:len(china)]
    germany = location_2d[len(china):len(china)+len(germany)]
    japan = location_2d[len(china)+len(germany):]

    server = akamai_ghost + apache + router
    server_2d = pca(server)

    akamai_ghost = server_2d[:len(akamai_ghost)]
    apache = server_2d[len(akamai_ghost):len(akamai_ghost) + len(apache)]
    router = server_2d[len(akamai_ghost) + len(apache):]

    certificate = trusted + untrusted
    certificate_2d = pca(certificate)

    trusted = certificate_2d[:len(trusted)]
    untrusted = certificate_2d[len(trusted):]

    print("ploting....")

    plt.figure(figsize = (24, 8))

    plt.subplot(1, 3, 1)
    plt.title("location")
    plt.scatter(china[:, 0], china[:, 1], color='r', label='China')
    plt.scatter(germany[:, 0], germany[:, 1], color = 'b', label='Germany')
    plt.scatter(japan[:, 0], japan[:, 1], color = 'y', label='Japan')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("servers")
    plt.scatter(akamai_ghost[:, 0], akamai_ghost[:, 1], color = 'r', label = 'AkamaiGHost')
    plt.scatter(apache[:, 0], apache[:, 1], color = 'b', label = 'Apache')
    plt.scatter(router[:, 0], router[:, 1], color = 'y', label = 'Router')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("certificate")
    plt.scatter(trusted[:, 0], trusted[:, 1], color = 'r', label = 'Trusted certificate')
    plt.scatter(untrusted[:, 0], untrusted[:, 1], color = 'b', label = 'Untrusted certificate')
    plt.legend()

    plt.savefig(os.path.join(SAVE_PATH, "cluster.png"))


if __name__ == '__main__':
    main()
