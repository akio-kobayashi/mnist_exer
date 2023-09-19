import torch
import pytorch_lightning as pl
from model import Solver
from digit_dataset import ImageTransform
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')
from torchvision import datasets, transforms
import os, sys, glob, re
import numpy as np
import matplotlib as plt

np.set_printoptions(precision=3, suppress=True)

def main(config:dict, checkpoint_path=None):
    model = Solver.load_from_checkpoint(checkpoint_path, config=config)

    out_channels = config['out_channels']
    if out_channels <= 8:
        row = 4
        col = 2
    elif out_channels <= 16:
        row = 4
        col = 4
    elif out_channels <= 24:
        row = 4
        col = 6
    elif out_channels <= 32:
        row = 4
        col = 8
    kernel_size = config['kernel_size']
    plt.figure(figsize=(16,16))

    save_dir = os.path.join(config['logger']['save_dir'], config['logger']['name'])
    save_dir = os.path.join(save_dir, 'version_' + str(config['logger']['version']))

    for num_layer in range(config['num_layers']):
        pattern = re.compile('convs.'+str(num_layer)+'0.weight')
        for key in model.model.state_dict().keys():
            if pattern.match(key):
                for i in range(out_channels):
                    kernel_weight = np.array(model.state_dict()[key])[i].reshape(kernel_size,kernel_size)
                    plt.subplot(row, col, i+1)
                    plt.title(f'フィルター係数 {num_layer} 層目 第 {i+1} フィルター')
                    plt.imshow(kernel_weight, cmap='gray')
                    outpath = os.path.join(save_dir, 'conv_'+ str(num_layer+1)+ '.png')
                    plt.savefig(outpath)
                                
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
    main(config, args.checkpoint)
