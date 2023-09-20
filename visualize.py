import torch
import pytorch_lightning as pl
from mnist_exer.model import Solver
from digit_dataset import ImageTransform
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')
from torchvision import datasets, transforms
import os, sys, glob, re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

np.set_printoptions(precision=3, suppress=True)

def visualize(config:dict, checkpoint_path=None):
    model = Solver.load_from_checkpoint(checkpoint_path, config=config).cpu()

    out_channels = config['num_filters']
    kernel_size = config['kernel_size']
    plt.figure(figsize=(16,16))

    save_dir = os.path.join(config['logger']['save_dir'], config['logger']['name'])
    save_dir = os.path.join(save_dir, 'version_' + str(config['logger']['version']))

    in_channels = 1
    for num_layer in range(config['num_layers']):
        pattern = re.compile('convs.'+str(num_layer)+'.0.weight')
        for key in model.model.state_dict().keys():
            if pattern.match(key):
                kernel_weight = np.array(model.model.state_dict()[key])
                for i in range(in_channels):
                    for j in range(out_channels):
                        kw = np.array(kernel_weight[j, i, :, :]) #np.array(model.model.state_dict()[key])[i][j]#.reshape(kernel_size,kernel_size)
                        plt.subplot(in_channels, out_channels, i*out_channels+j+1)
                        plt.title(f'{num_layer+1} 層目 {i*out_channels + j +1} フィルター')
                        plt.imshow(kw, cmap='gray')
                        outpath = os.path.join(save_dir, 'conv_'+ str(num_layer+1)+ '.png')
                        plt.savefig(outpath)
        in_channels = out_channels
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    if 'config' in config.keys():
        config = config['config']
    visualize(config, args.checkpoint)
