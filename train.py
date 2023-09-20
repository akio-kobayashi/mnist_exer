import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
import torch.utils.data as dat
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')
from torchvision import datasets, transforms
from mnist_exer.model import Solver

def main(config:dict, checkpoint_path=None):
    #model = Solver.load_from_checkpoint(checkpoint_path, config=config)
    model = Solver(config)
    
    # MNIST
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        config['mnist_save_dir'],
        train = True,      
        download = True,   
        transform = transform
    )
    valid_dataset = datasets.MNIST(
        config['mnist_save_dir'], 
        train = False,
        transform = transform
    )

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
    parser.add_argument('--checkpoint', type=str, default=None)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config, args.checkpoint)
