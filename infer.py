import torch
import pytorch_lightning as pl
from mnist_exer.model import Solver
from mnist_exer.digit_dataset import ImageTransform
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')
from torchvision import datasets, transforms
import os, sys, glob, re
import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

def ToDataframe(file, idx, probs):
    keys = ['file', 'infer', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    probs = probs.tolist()
    print(len(probs))
    item = {}
    for n in range(len(keys)):
        if n == 0:
            item[keys[n]] = file
        elif n == 1:
            item[keys[n]] = idx
        else:
            item[keys[n]] = probs[n-2]
    return pd.DataFrame(item)

def infer(config:dict, checkpoint_path=None, dirs=None):
    model = Solver.load_from_checkpoint(checkpoint_path, strict=False, config=config)
    transformer = ImageTransform()

    df = None
    #results={}
    num_correct = 0
    num_test = 0
    for path in glob.glob(os.path.join(dirs, '**/*.png'), recursive=True):
        image, label = transformer(path)
        idx, probs = model.infer(image.cuda())
        print(len(probs.tolist()))
        #results[path] = {'index': idx, 'probs': probs}
        df_line = ToDataframe(path, idx, probs)
        #df_line = None
        if df is None:
            df = df_line
        else:
            df = pd.concat([df, df_line])
        if idx == label:
            num_correct += 1
        num_test += 1
    rate = num_correct/num_test * 100.
    #print(f'正解率: {rate} %')
    html = df.to_html()
    #results_sort = sorted(results.keys())
    #for path in results_sort:
    #    print(f'{path} 推定: {results[path]["index"]} 確率: {results[path]["probs"]}')
    return rate, html

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
    infer(config, args.checkpoint)
