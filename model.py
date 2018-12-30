from clize import run
from torch.nn import init
from torch.nn.init import xavier_uniform_
import torch
import time
import os
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from data import Dataset

class Sin(nn.Module):

    def forward(self, x):
        return torch.sin(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)

class VAE_CPPN(nn.Module):
    def __init__(self, latent_size=100, ensemble_dim=10, output_dim=10):
        super().__init__()
        self.latent_size = latent_size
        self.ensemble_dim = ensemble_dim
        self.encode = nn.Sequential(
            nn.Conv1d(1, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(128, latent_size * 2, 4, 2, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.decode = nn.Sequential(
            nn.Linear(latent_size * 2, 1024),
            nn.Tanh(),
            nn.Linear(1024, ensemble_dim),
            nn.Tanh(),
        )
        self.enhance = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv1d(128, 1, 3, 1, 1),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        h = self.encode(x)
        h = h.view(h.size(0), h.size(1))
        mu, logvar = h[:, 0:self.latent_size], h[:, self.latent_size:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        h = mu# + eps * std
        t = torch.linspace(-1, 1, x.size(2))
        device = next(self.parameters()).device
        t = t.to(device)
        l = h.view(h.size(0), 1, h.size(1)).expand(h.size(0), x.size(2), h.size(1))
        t = t.view(1, -1, 1)
        t = t.expand(h.size(0), t.size(1), h.size(1))
        z = torch.cat((l, t), dim=2)
        z_ = z.view(-1, self.latent_size * 2) 
        xrec = self.decode(z_)
        xrec = xrec.view(z.size(0), self.ensemble_dim, z.size(1))
        xrec = xrec.mean(dim=1, keepdim=True)
        #xrec = self.enhance(xrec)
        return xrec, mu, logvar

    def loss_function(self, x, xrec, mu, logvar):
        x = x.view(x.size(0), -1)
        xrec = xrec.view(xrec.size(0), -1)
        mse = ((xrec - x) ** 2).sum(1).mean()
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return mse + kld



class VAE(nn.Module):
    def __init__(self, latent_size=100, output_dim=10, ensemble_dim=None):
        super().__init__()
        self.latent_size = latent_size
        self.encode = nn.Sequential(
            nn.Conv1d(1, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(128, latent_size * 2, 4, 2, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.decode = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        h = self.encode(x)
        h = h.view(h.size(0), h.size(1))
        mu, logvar = h[:, 0:self.latent_size], h[:, self.latent_size:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        h = mu + eps * std
        xrec = self.decode(h)
        xrec = xrec.view(x.size())
        return xrec, mu, logvar

    def loss_function(self, x, xrec, mu, logvar):
        x = x.view(x.size(0), -1)
        xrec = xrec.view(xrec.size(0), -1)
        mse = ((xrec - x) ** 2).sum(1).mean()
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return mse + kld
if __name__ == '__main__':
    ae = VAE()
    x = torch.randn(32, 1, 16000)
    xrec, mu, logvar = ae(x)
    print(xrec.size(), mu.size(), logvar.size())
