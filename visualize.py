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

np.set_printoptions(precision=3, suppress=True)

def main(config:dict, checkpoint_path=None):
    model = Solver.load_from_checkpoint(checkpoint_path, config=config)

    pattern = re.compile('convs.\d+.\d+.weight')

    for key in model.model.state_dict().keys():
        if pattern.match(key):
            print(key)
    
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
