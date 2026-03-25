import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn as nn
import torch
import lightning.pytorch as pl
from torchmetrics import Accuracy
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics import Accuracy, Recall, Precision, F1Score, ConfusionMatrix, AUROC
from torchmetrics.classification import CohenKappa
from MAE_pretraining.data_lightning import EEGData
import lightning as L
from lightning.pytorch import Trainer
import torch.nn.functional as F
from lightning.pytorch.callbacks import TQDMProgressBar
import math
from MAE_pretraining.graph_embedding import GraphDataset
from MAE_pretraining.gnn import GATModel
import random
import os
import torch.nn.init as init
from timm.models.vision_transformer import PatchEmbed, Block
from torch_geometric.data import Batch
from MAE_pretraining.transformer_variants import TransformerLayerViT


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


class TemporalEncoding(nn.Module):
    """Sinusoidal positional embedding implementation"""
    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        #Arange all position in the sequence dim = (max_len, 1)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  

        #Arange exponential along the embedding dimension
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        )  # (D/2,)

        #Create a matrix of sequence length against temporal embedding and assign frequency to each entry
        encoding = torch.zeros(1, max_len, embed_dim, dtype=torch.float32)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("encoding", encoding)

    def get_class_token(self):
        return self.encoding[0,0,:]

    def forward(self, seq_length: int, num_channel: int) -> torch.Tensor:
        if seq_length > self.max_len:
            raise ValueError(f"seq_length={seq_length} exceeds max_len={self.max_len}")
        
        #Select the embedding size corresponding to the sequence length
        pe = self.encoding[:, :seq_length, :]               # (1, Seq, D)
        #Repeat it over channels
        pe = pe.repeat_interleave(num_channel, dim=1)       # (1, Seq*C, D)
        return pe
    

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


class EncoderDecoder(pl.LightningModule):
    """Basic encoder decoder model following the ViT model"""
    def __init__(self, config = None, use_rotary = False,num_channels = 64, 
                 max_embedding = 2000, enc_dim = 512, dec_dim = 384, depth_e = 8, 
                 depth_d = 4, mask_prob = 0.7, patch_size = 16, norm_pix_loss = False):
        super().__init__()

        self.config = config
        self.use_rotary = use_rotary
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim


        #Define the encoder and decoder layers
        self.encoder = nn.ModuleList([TransformerLayerViT(enc_dim, nhead=8, mlp_ratio=4, qkv_bias=True, norm=nn.LayerNorm) for i in range(depth_e)])
        self.decoder = nn.ModuleList([TransformerLayerViT(dec_dim, nhead=16, mlp_ratio = 4, qkv_bias=True, norm=nn.LayerNorm) for i in range(depth_d)])

        #Set the probability for a token to be masked
        self.mask_prob = mask_prob

        #Layer convert the encoder dimension to the decoder dimension
        self.encoder_decoder = (nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity())

        self.patch_size = patch_size
        self.patch = PatchEEG(patch_size=patch_size, embed_dim=enc_dim)
        self.mask_token = nn.Parameter(torch.zeros((1, 1, dec_dim)))
        self.fc = nn.Linear(dec_dim, patch_size)
        self.num_channels = num_channels
        self.norm_enc = nn.LayerNorm(enc_dim)
        self.norm_dec = nn.LayerNorm(dec_dim)
        
        #Define the temporal and channel embeddings 
        self.channel_embedding_e = ChannelPositionalEmbed(embedding_dim=enc_dim)
        self.channel_embedding_d = ChannelPositionalEmbed(embedding_dim=dec_dim)
        self.temporal_embedding_d = TemporalPositionalEncoding(d_model=dec_dim, max_len=max_embedding)
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
        nn.init.normal_(self.channel_embedding_d.channel_transformation.weight, std=0.02)


    def restore_seq(self, x, num_patches, id_restore):

        B, L_keep, D = x.shape
        x_full = self.mask_token.repeat(B,num_patches - L_keep,1).clone()
        x = torch.cat([x,x_full], dim=1)
        x = torch.gather(x, dim=1, index=id_restore.unsqueeze(-1).repeat(1,1,D))
        return x

        

    def to_decoded(self, x, num_patches, keep_id):
        """Gets back the original ordering of patches"""
        B, L_keep, D = x.shape
        device = x.device

        #Get the total number of patches 
        L = num_patches

        #Get the ids of the unmasked tokens
        keep_id = keep_id.unsqueeze(-1).expand(B, -1, x.shape[-1])

        #Get full sized random patches and add the kept token in their original positions
        x_full = self.mask_token.view(1,1,-1).expand(B,L,D).clone()
        x_full.scatter_(dim = 1, index=keep_id, src = x)
        return x_full
    

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    

    def masking_vit(self, x):
        B, L, D = x.shape
        device = x.device
        noise = torch.rand(B,L, device=device)
        l_keep = int(L*(1-self.mask_prob))
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        id_keep = ids_shuffle[:,:l_keep]
        id_keep = repeat(id_keep, "b n -> b n d", d = D)
        x_masked = torch.gather(x, dim=1, index=id_keep)
        mask = torch.ones([B,L], device=device)
        mask[:,:l_keep] = 0
        mask = torch.gather(mask, dim = 1, index=ids_restore)


        return x_masked, ids_restore,mask



    def masking(self, x):
        """Mask random patch of the eeg input"""

        B, N, C, T = x.shape
        device = x.device

        while True:
            chan_drop = [c for c in range(C) if random.random() < 0.2]
            if len(chan_drop) < C:
                break
        
        #Pick the channels to drop
        if len(chan_drop) > 0:
            idx_chan = torch.arange(0,N, device=device, dtype=torch.long).repeat_interleave(len(chan_drop))
            idx_chan = idx_chan.reshape(N,(len(chan_drop))) * C + torch.tensor(chan_drop, device=device, dtype=torch.long)
            idx_chan = idx_chan.view(-1)
        else:
            idx_chan = torch.empty(0, device=device, dtype=torch.long)

        #Compute total number of patch
        L = N*C

        #Stack all patches in a 1D array
        x_flat = rearrange(x, "b n c t -> b (n c) t")
        
        #Compute the number of patches to keep
        num_drop = idx_chan.numel()
        num_allowed = L - num_drop
        L_keep = min(int(L * (1 - self.mask_prob)), num_allowed)

        #Retrieve shuffle as well as ordered indices
        noise = torch.rand(B,L, device=device)
        noise[:,idx_chan] = 2
        id_shuffle = torch.argsort(noise, dim = 1)
        #id_restore = torch.argsort(id_shuffle, dim = 1)

        #Retrieve the id that we will be kept, sort them and repeat it T times so that patches are either fully
        # visible or fully masked
        id_keep = id_shuffle[:, :L_keep].to(device=device)    
        id_sorted, _ = torch.sort(id_keep, dim=1)
        to_keep = repeat(id_sorted, "b l -> b l t", t = T)

        #Only keep the non masked indices from the original signal
        x_return = torch.gather(x_flat, dim = 1, index=to_keep)
        
        #Compute the mask to remove the non masked patches for prediction
        mask = torch.ones(B,L, device=device)
        mask.scatter_(dim = 1, index=id_sorted, src=torch.zeros(B,L_keep, dtype=mask.dtype, device=device))

        return x_return, mask, id_sorted
    
    def encoder_forward(self,x, channel_list):
        B, C, T = x.shape
        device = x.device
        channel_list = torch.tensor(channel_list, dtype=torch.long, device=device)
    
        #Return the patch eeg with shape (b, n, c, d)
        x = self.patch(x)
        original = x
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

        if not self.use_rotary:
            seq_idx = torch.arange(0, N, device=device, dtype=torch.long)  # use 0..Seq-1 (or 1..Seq if your ref does)
            eeg_seq_indices = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)

            tp = self.temporal_embedding_e(eeg_seq_indices) 
            x += tp

        #Mask the eeg patches
        x, ids_restore, mask = self.masking_vit(x)

        #Concatenate the class token to the eeg
        class_token = self.class_token + self.temporal_embedding_e.get_class_token()
        class_token = class_token.expand(B,1,-1)
        x = torch.cat([class_token, x], dim = 1)
       
        #Pass the sequence through the encoder layers 
        for transformer in self.encoder:
            x = transformer(x)

        x = self.norm_enc(x)

        return x, ids_restore, mask, original

    def decoder_forward(self, x, ids_restore, mask, original, channel_list):
        B, N, C, D = original.shape
        L = N*C
        device = x.device
        #Get decoder dimensions (B,N*C,D)
        x = self.encoder_decoder(x)
        #Retrieve the class token which is not used for decoding
        x, class_token = x[:,1:,:], x[:,:1,:]

        
        #Get the original ordering of patches
        x = self.restore_seq(x = x, num_patches=L, id_restore=ids_restore)
        

        #Get embeddings for decoding
       
       
        if channel_list.dim() == 1:
            channel_list = channel_list.unsqueeze(0).expand(B, -1) # Make it (B, C)
            
        # (B, C) -> (B, 1, C) -> (B, N, C) -> (B, N*C)
        chan_id = channel_list.unsqueeze(1).repeat(1, N, 1).view(B, L)
        chan_embedding = self.channel_embedding_d(chan_id)
        
        x = x + chan_embedding

        #Add the temporal embedding if the rotary embedding are not used
        if not self.use_rotary:
            seq_idx = torch.arange(0, N, device=device, dtype=torch.long)  # use 0..Seq-1 (or 1..Seq if your ref does)
            eeg_seq_indices = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)

            tp = self.temporal_embedding_d(eeg_seq_indices) 
            x = x + tp

        
        class_token = class_token + self.temporal_embedding_d.get_class_token()
        x = torch.cat([class_token, x], dim = 1)

        #Get the eeg sample indices for rotary embeddings
        seq = torch.arange(0,N).repeat_interleave(C)
        seq = seq.expand(B, -1)
        #Pass input through decoder layers
        for transformer in self.decoder:
            x = transformer(x)
        
        #Only keep the initially masked patches
        x = x[:,1:,:]
        #x = x[torch.where(mask == 1)]

        x = self.norm_dec(x)
        x = self.fc(x)

        return x, mask

    def forward(self, eeg, channel_list):

        x, ids_restore, mask, original = self.encoder_forward(eeg, channel_list)
        pred, mask = self.decoder_forward(x, ids_restore, mask, original, channel_list)
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

    
    def create_batch_graphs(self,eeg_batch):
        graph_list = []
        for eeg, chn_list in eeg_batch:
            g = self.graph_gen.create_graph(chn_list)
            graph_list.append(g)
        graph_batch = Batch(graph_list)
        return graph_batch


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
        

    def test_step(self, batch, batch_idx):
        data, target = batch
        pred = self(data)
        loss = self.criterion(pred, target)
        self.log("test_loss", loss)

    def predict_step(self):
        pass





if __name__ == "__main__":
    seed_everything(42)
    L.seed_everything(42, workers=True)

    model = EncoderDecoder()
    #data = EEGData(data_dir="MAE_pretraining/data_bis")
    ckpt = ModelCheckpoint(
    monitor="val_mse",
    mode="min",
    save_top_k=1,
    filename="mae-{epoch:02d}-{val_mse:.4f}",
)
    #early = EarlyStopping(monitor="val_mse", mode="min", patience=10)
    #trainer = Trainer(callbacks=[TQDMProgressBar(refresh_rate=20), ckpt, early], log_every_n_steps=5, max_epochs=15)
    #trainer.fit(model, val_dataloaders=valid_loader, train_dataloaders=train_loader)
    tensor_test = torch.rand(3,100,32,250)
    model.masking(tensor_test)
