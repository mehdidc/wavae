import matplotlib as mpl
mpl.use('Agg')
from scipy.io import wavfile
import matplotlib.pyplot as plt
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import model
from transforms import Scale, Compose, PadTrim
from data import Dataset
from clize import run
def train(
        *,
        data_folder='data',
        nb=None,
        lr=1e-3,
        model_name='VAE_CPPN',
        batch_size=32, 
        epochs=1000,
        input_dim=1,
        max_len=500,
        log_interval=50,
        latent_size=10,
        ensemble_dim=1,
        cuda=False):
    mod = getattr(model, model_name)
    vae = mod(latent_size=latent_size, output_dim=max_len, ensemble_dim=ensemble_dim)
    if cuda:
        vae = vae.cuda()
    optimizer = optim.Adam(
        vae.parameters(), lr=lr,
    )
    vae.train()
    epoch_start = 1
    transform = Compose([
        PadTrim(max_len=max_len),
        Scale(),
    ])
    if nb:
        nb = int(nb)
    dataset = Dataset(data_folder, transform=transform, nb=nb)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    nb_iter = 0
    for epoch in range(epoch_start, epochs+1):
        for batch_index, data in enumerate(dataloader):
            x = data
            x = x.cuda() if cuda else x
            vae.zero_grad()
            xrec, mu, logvar = vae(x)
            loss = vae.loss_function(x, xrec, mu, logvar)
            loss.backward()
            optimizer.step()
            if nb_iter % log_interval == 0:
                print(f'niter: {nb_iter:05d} loss: {loss.item():.4f}')
                x = x.detach().cpu().numpy()
                xrec = xrec.detach().cpu().numpy()
                signal = x[0:3, 0].T
                fake_signal = xrec[0:3, 0].T
                for i in range(len(xrec)):
                    s = xrec[i, 0]
                    wavfile.write(f'out/fake_{i:03d}.wav', 16000, s)
                for i in range(len(x)):
                    s = x[i, 0]
                    wavfile.write(f'out/real_{i:03d}.wav', 16000, s)
                fig = plt.figure(figsize=(50, 10))
                plt.plot(signal, color='blue', label='true')
                plt.plot(fake_signal, color='orange', label='fake')
                #plt.legend()
                plt.savefig('out.png')
                plt.close(fig)
            nb_iter += 1

if __name__ == '__main__':
    run(train)
