import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=256, hidden_dim=1024):
        super(VAE, self).__init__()

        self.fc_e = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.fc_d1 = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.fc_d2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.input_dim = input_dim

    def encoder(self, x_in):
        x = F.relu(self.fc_e(x_in))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def decoder(self, z):
        z1 = F.relu(self.fc_d1(z))
        x_out = torch.sigmoid(self.fc_d2(z1))
        return x_out

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar * 0.5)
        e = Variable(torch.randn(sd.size())).cuda()
        z = e.mul(sd).add_(mean)
        return z

    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar


def loss_func(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    LOSS = BCE + KLD
    return LOSS
