import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from mnist_exer.model import Solver
import torch.utils.data as dat
from mnist_exer.digit_dataset import DigitDataset
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')
from torchvision import datasets, transforms

def finetune(config, checkpoint_path, train_datadir, valid_datadir):
    config['logger']['name'] = 'finetune'
    config['checkpoint']['filename'] = 'finetune'
    config['checkpoint']['save_last'] = True
    config['optimizer']['lr'] = 1.e-6
    config['num_batch'] = 4
    config['trainer']['max_epochs'] = 100
    print(config)
    
    model = Solver(config)
    
    # Handwritten Digits
    train_dataset = DigitDataset(train_datadir)
    valid_dataset = DigitDataset(valid_datadir)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config['num_batch'],
        shuffle = True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,     
        batch_size = config['num_batch'],
        shuffle = True
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    ]

    logger = TensorBoardLogger(**config['logger'])
    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          **config['trainer'] )
    trainer.fit(model=model, ckpt_path=args.checkpoint,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--valid_dir', type=str, required=True)
    
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    if 'config' in config.keys():
        config = config['config']
        
    finetune(config, args.checkpoint, args.train_dir, args.valid_dir)
