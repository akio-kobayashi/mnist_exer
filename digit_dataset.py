import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from torch import Tensor
import torchvision
import torchvision.io as IO
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import glob

class ImageTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize(28),
            T.ToTensor()
        ])

    def forward(self, path):
        image = Image.open(os.path.abspath(path))
        image = self.transform(image)
        label = int(list(path.replace('.png', ''))[-1])
        return image, label
    
class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, datadir):
        super().__init__()
        self.paths=[]
        self.labels=[]
        for path in glob.glob(os.path.join(datadir, '**/*.png'), recursive=True):
            self.paths.append(os.path.abspath(path))
            lab = int(list(path.replace('.png', ''))[-1])
            assert lab >= 0 and lab <=9
            self.labels.append(lab)
        self.transform = T.Compose([
            T.Grayscale(), 
            T.RandomHorizontalFlip(),
            T.RandomPerspective(),
            T.RandomRotation(10),
            T.Resize(28), T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path)
        image = self.transform(image)
        label = self.labels[idx]

        return image, label
    
