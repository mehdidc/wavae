import torch
from torch.utils.data import DataLoader
from scipy.io import wavfile
from transforms import Scale, Compose, PadTrim
import os
from glob import glob

class Dataset:

    def __init__(self, folder, transform=None, nb=None):
        self.folder = folder
        self.classes = os.listdir(folder)
        self.filenames = glob(os.path.join(folder, '**', '*.wav'))
        if nb:
            self.filenames = self.filenames[0:nb]
        self.transform = transform
    
    def __getitem__(self, idx):
        sample_rate, signal = wavfile.read(self.filenames[idx])
        signal =  signal.copy()
        signal = torch.from_numpy(signal)
        signal = signal.view(1, -1)
        if self.transform:
            signal = self.transform(signal)
        return signal
    
    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    transform = Compose([
        Scale(),
        PadTrim(max_len=16000),
    ])
    dataset = Dataset('data', transform=transform)
    print(dataset[0].size())
    dataloader = DataLoader(dataset, batch_size=32)
