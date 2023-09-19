import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

class Solver(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.model = mnistModel(config)
        self.ce_loss = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, x):
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
    
    def infer(self, x):
        if x.dim() == 3:
            x = rearrange(x, '(b c) h w -> b c h w', b=1)
        logits = self.forward(x)
        probs = F.softmax(logits).detach().numpy()[0]

        idx = torch.argmax(dim=-1).detach().numpy()

        return idx, probs

class LayerNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')
    
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
                LayerNorm(out_channels)
            ]
            self.convs.append(nn.Sequential(*block))
            in_channels = out_channels
            valid_width = int ( ( valid_width + 2*(config['kernel_size']//2) - config['kernel_size'] ) / config['stride'] + 1 )

        self.feedforward = nn.Linear(valid_width*valid_width*out_channels, 10)

    def forward(self, x):
        for block in self.convs:
            x = block(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        return self.feedforward(x)
    
