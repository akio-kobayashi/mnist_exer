import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

class Solver():
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.model = mnistModel(config)
        self.ce_loss = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def foward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_index):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.ce_loss(logits, labels)
        self.log_dict({'train_loss': loss})

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.ce_loss(logits, labels)
        self.log_dict({'valid_loss': loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.config['optimizer'])
        return optimizer
    
class mnistModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = 1
        out_channels = config['num_filters']

        valid_width=28
        self.convs = nn.ModuleList()
        for n in range(config['num_layers']):
            block = [ ]
            block += [
                nn.Conv2d(in_channels, 
                          out_channels, 
                          config['kernel_size'], 
                          config['stride'], 
                          padding=config['kernel_size']//2
                          ),
                nn.ReLU(),
                nn.BatchNorm()
            ]
            self.convs.append(nn.Sequential(*block))
            in_channels = out_channels
            valid_width = int ( ( valid_width + 2*config['kernel_size']//2 - config['kernel_size'] ) / config['stride'] + 1 )

        self.feedforward = nn.Linear(valid_width*valid_width*out_channels, 10)

    def forward(self, x):
        x = self.convs(x)
        x = rearrange(x, 'b c f -> b (c f)')
        return self.feedforward(x)
    