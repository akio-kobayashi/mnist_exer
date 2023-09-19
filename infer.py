import torch
import pytorch_lightning as pl
from model import Solver
from digit_dataset import ImageTransform
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')
from torchvision import datasets, transforms
import os, sys, glob
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def main(config:dict, checkpoint_path=None):
    model = Solver.load_from_checkpoint(checkpoint_path, config=config)
    transformer = ImageTransform()

    for path in glob.glob(os.path.join(args.dir, '**/*.png'), recursive=True):
        image, label = transformer(path)
        idx, probs = model.infer(image.cuda())
        print(f'{path} 推定: {idx} 確率: {probs}')

    print(model.model.state_dict().keys())
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    if 'config' in config.keys():
        config = config['config']
    main(config, args.checkpoint)
