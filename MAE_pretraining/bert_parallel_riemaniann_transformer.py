import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from MAE_pretraining.data_lightning import EEGData
import lightning as L
from lightning.pytorch import Trainer
import torch.nn.functional as F
from lightning.pytorch.callbacks import TQDMProgressBar
import math
import random
import os
import torch.nn.init as init
from MAE_pretraining.transformer_variants import RiemannianParallelCrissCrossTransformer


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


channel_list = ["Fp1","Fp2","AF3","AF4","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","PO7","PO3","PO4","PO8","Oz",]
channel_list_2 = ["Fp2","AF3","AF4","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","PO7","PO3","PO4","PO8","Oz",]
CHANNEL_DICT = dict(zip(range(len(channel_list)), channel_list))


class PatchEEG(nn.Module):
    """Module that segments the eeg signal in patches and align them with encoder dimension"""
    def __init__(self, embed_dim = 768, patch_size = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(patch_size, embed_dim)
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=(1,patch_size), stride=patch_size)
    def patch_eeg(self, x):
        B, C, T = x.shape

        #Add one dimension B,C,1,T
        x = x.unsqueeze(2)
        #size is (B, C*16, seq)
        #Retrieves patch of size patch_size for every channel and stack them 
        x = self.unfold(x)
        x = rearrange(x, "b (c p) s -> b c p s", c = C, p = self.patch_size)
        #Shape of B seq C p with p being patch size
        x = x.permute(0, 3, 1, 2)
        return x
    def forward(self, x):
        #Patch the eeg and project it to the encoder dimension
        x = self.patch_eeg(x)
        return self.fc(x)
    

    
class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim):
        super(ChannelPositionalEmbed, self).__init__()
        self.channel_transformation = nn.Embedding(144, embedding_dim)
        init.zeros_(self.channel_transformation.weight)
    def forward(self, channel_indices):
        channel_embeddings = self.channel_transformation(channel_indices)
        return channel_embeddings


    

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float())
        pe = torch.zeros(1, max_len, d_model)
        pe[0,:, 0::2] = torch.sin(position.float() * div_term)
        pe[0,:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)
    
    def get_class_token(self):
        return self.pe[0,0,:]
    
    def forward(self, seq_indices):
        batch_size, seq_len = seq_indices.shape
        pe_embeddings = self.pe[0, seq_indices.view(-1)].view(batch_size, seq_len, -1)
        return pe_embeddings


class RiemaniannTransformerBert(pl.LightningModule):
    """Basic encoder decoder model following the ViT model"""
    def __init__(self, config = None, num_channels = 64, 
                 max_embedding = 2000, enc_dim = 512, depth_e = 8, 
                  mask_prob = 0.5, patch_size = 16, norm_pix_loss = False):
        super().__init__()

        self.config = config
        self.enc_dim = enc_dim


        #Define the encoder layers (CBraMod-style parallel head-split)
        self.encoder = nn.ModuleList([RiemannianParallelCrissCrossTransformer(enc_dim, nhead=8, mlp_ratio=4) for i in range(depth_e)])
        #Set the probability for a token to be masked
        self.mask_prob = mask_prob

        self.patch_size = patch_size
        self.patch = PatchEEG(patch_size=patch_size, embed_dim=enc_dim)
        self.mask_token = nn.Parameter(torch.zeros((1, 1, enc_dim)))
        self.fc = nn.Linear(enc_dim, patch_size)
        self.num_channels = num_channels
        self.norm_enc = nn.LayerNorm(enc_dim)
        
        #Define the temporal and channel embeddings 
        self.channel_embedding_e = ChannelPositionalEmbed(embedding_dim=enc_dim)
        self.temporal_embedding_e = TemporalPositionalEncoding(d_model=enc_dim, max_len=max_embedding)

        self.criterion = nn.MSELoss()
        self.class_token = nn.Parameter(torch.zeros(1,1,enc_dim))
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        self.apply(self._init_weights)

        # Re-do patch projection explicitly
        w = self.patch.fc.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        nn.init.normal_(self.channel_embedding_e.channel_transformation.weight, std=0.02)
      


    def mask_bert(self, x):
        device = x.device
        B, L, D = x.shape
        mask = torch.rand(B,L, device=device) < self.mask_prob
        mask_token = self.mask_token.expand_as(x)
        mask_float = mask.unsqueeze(-1).float()
        x = x * (1 - mask_float) + mask_token * mask_float
        return x, mask

    
    def encoder_forward(self,x, channel_list):
        B, C, T = x.shape
        device = x.device
        channel_list = torch.tensor(channel_list, dtype=torch.long, device=device) if not isinstance(channel_list, torch.Tensor) else channel_list.to(device)

        #Return the patch eeg with shape (b, n, c, d)
        x = self.patch(x)
        N = x.shape[1]
        L = x.shape[1] * x.shape[2]
        x = x.reshape(B,L,-1)

        #Define the embeddings
        if channel_list.dim() == 1:
            channel_list = channel_list.unsqueeze(0).expand(B, -1) # Make it (B, C)
            
        # (B, C) -> (B, 1, C) -> (B, N, C) -> (B, N*C)
        chan_id = channel_list.unsqueeze(1).repeat(1, N, 1).view(B, L)
        chan_embedding = self.channel_embedding_e(chan_id)
        
        x += chan_embedding 

       
        seq_idx = torch.arange(0, N, device=device, dtype=torch.long)  # use 0..Seq-1 (or 1..Seq if your ref does)
        eeg_seq_indices = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)

        tp = self.temporal_embedding_e(eeg_seq_indices) 
        x += tp

        #Mask the eeg patches
        x, mask = self.mask_bert(x)

        #Pass the sequence through the encoder layers (parallel head-split per layer)
        for transformer in self.encoder:
            x = transformer(x, C)

        x = self.norm_enc(x)
        x = self.fc(x)

        return x, mask

    

    def forward(self, eeg, channel_list):

        pred, mask = self.encoder_forward(eeg, channel_list)
        target, pad = self.patchify_1d(eeg, self.patch_size)   
        B, Seq, Ch, P = target.shape
        target = target.view(B, Seq * Ch, P)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, unbiased=False, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5                 
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)    # (B, L)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, pred, mask



    def patchify_1d(self, x, patch_size: int):
        """Segment the target eeg signal in patch and pad if necessary"""
        B, C, T = x.shape
        #Get the padding needed
        pad = (patch_size - (T % patch_size)) % patch_size

        #Pad the temporal dimension if neededs
        if pad > 0:
            x = F.pad(x, (0, pad))  
            T = T + pad

        #Return the result in a patched form
        N = T // patch_size
        x = x.view(B, C, N, patch_size)       
        x = x.permute(0, 2, 1, 3).contiguous()
        return x, pad 
    
    def configure_optimizers(self):
        # 1. Use AdamW with Weight Decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)
        
        # 2. Setup Cosine Annealing with Warmup
        # Assuming args.epochs = 500 and a standard batch schedule
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,              # Peak learning rate
            total_steps=total_steps,  # Total batches across all epochs
            pct_start=0.15,            # 10% of training spent warming up (e.g., 50 epochs)
            anneal_strategy='cos',    # Cosine decay
            div_factor=10.0,          # Start LR = max_lr / 10
            final_div_factor=1000.0   # End LR = start_lr / 1000
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Update the LR after every batch, not epoch
            }
        }
    
    def training_step(self, batch, batch_idx):
        data, channel_list = batch
        loss, pred, mask = self(data, channel_list)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step = True, prog_bar = False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, channel_list = batch
        mse, pred, mask = self(data, channel_list)
        
        rmse = torch.sqrt(mse + 1e-8)
        pred_std = pred.std()

        self.log_dict(
        {"val_mse": mse, "val_rmse": rmse, "val_pred_std": pred_std},
        prog_bar=True, on_step=False, on_epoch=True
    )
        





if __name__ == "__main__":
    seed_everything(42)
    L.seed_everything(42, workers=True)

   