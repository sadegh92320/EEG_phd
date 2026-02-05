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
import random

channel_list = ["Fp1","Fp2","AF3","AF4","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","PO7","PO3","PO4","PO8","Oz",]
channel_list_2 = ["Fp2","AF3","AF4","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","PO7","PO3","PO4","PO8","Oz",]
CHANNEL_DICT = dict(zip(range(len(channel_list)), channel_list))


class PatchEEG(nn.Module):
    """Module that segments the eeg signal in patches and align them with encoder dimension"""
    def __init__(self, embed_dim = 768, patch_size = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(patch_size, embed_dim)
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=(1,patch_size), stride=patch_size)
    def patch_eeg(self, x):
        B, C, T = x.shape

        #Pad the eeg signal so that it is aligned with the patch size
        pad = (self.patch_size - (T%self.patch_size))%self.patch_size
        if pad > 0:
            x = F.pad(x, (0,pad))

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
    
class DropPath(nn.Module):
    """Randomly drops the attention branch"""
    def __init__(self, keep_prob):
        super().__init__()
        self.prob = keep_prob
    
    def drop_path(self, x):
        #Deactivate if training
        if self.prob == 1 or self.training == False:
            return x
        
        #Dim of tensor B, 1, 1,...
        dim_tensor = (x.shape[0],) + ([1]) * (x.ndim - 1)

        #Create a tensor of size B,1,1... with values between 0 and 2
        rand_tensor = self.prob + torch.rand(dim_tensor, dtype=x.dtype, device=x.device)
        #Convert values to binary
        drop_tensor = rand_tensor.floor()
        #Drop the chosen values and divide by the probability to keep the same original expectation
        return x.div(self.prob) * drop_tensor
    
    def forward(self, x):
        return self.drop_path(x)

class RotaryEmbedding(nn.Module):
    """
        Implementation of the Rotary embedding which attributes to each token
        pair a relative rotation.
    """
    def __init__(self, model_dim,theta = 10000,is_learnable = False):
        super().__init__()
        assert model_dim % 2 == 0
        self.is_learnable = is_learnable

        #Define the frequency of the angle
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, model_dim, 2)[:(model_dim // 2)].float() / model_dim)),
            requires_grad = is_learnable)

    def create_frequency(self, kept_ids, device, dtype):

        #Total number of patches
        #Shape (num_patches)
        B, N = kept_ids.shape
        kept_ids = kept_ids.to(self.freqs.dtype).to(self.freqs.device)

        #Do the outer product between the freqs and all the kept patch indices
        freqs = kept_ids.reshape(B,N,1) * self.freqs
        #Convert the angle to polar coordinate
        polar = torch.polar(torch.ones_like(freqs), freqs)
        return polar

    def forward(self, x, id_kepts):
        #B, num_head, num_patch, dim_head
        B, H, N, D = x.shape
        device = x.device
        dtype = x.dtype
        ndim = x.ndim
        assert D == 2 * self.freqs.numel(), "RoPE model_dim must equal x last dim"

        assert id_kepts.shape == (B, N)
    
        polar = self.create_frequency(kept_ids=id_kepts, device=device, dtype=dtype)

        #Dimension B, 1, N, D
        polar = polar.unsqueeze(1)

        #Dimension B, num_head, N, D//2, 2
        #Segment the last dimension in individual 2D plan
        x2 = x.view(*x.shape[:-1], D//2, 2).contiguous()

        #Convert the vector is polar form to complex number
        x_complex = torch.view_as_complex(x2)

        #Multiply the vector by the angle then transform back to real number with correct shape
        x_rotate = x_complex * polar
        x_real = torch.view_as_real(x_rotate)
        x_real = x_real.reshape(*x.shape)
        return x_real

class MultiHeadAttention(nn.Module):
    """Multi head attention module, takes the embedding dim and number of head."""
    def __init__(self, embed_dim, num_heads = 3, dropout = 0.1, att_dropout = 0.1,is_causal = False, return_att = False, use_rotary = False, has_cls = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.dim_head = embed_dim//num_heads
        self.qkv = nn.Linear(embed_dim, 3*embed_dim)
        self.fc = nn.Linear(embed_dim,embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal
        self.return_att = return_att
        self.att_dropout = att_dropout
        self.use_rotary = use_rotary

        #Used as a flag to signal if the input is expected to have the class token
        self.has_cls = has_cls

        self.rotary = RotaryEmbedding(model_dim=embed_dim//num_heads)

    def split_heads(self, X):
        return X.view(X.size(0), X.size(1), 3, self.h, self.dim_head).permute(2, 0, 3, 1, 4)      

    def forward(self, x, position = None):
        #Compute Q, K and V and seperate segments per head
        B, N, D = x.shape

        #Extract the query key value vectors each with shape
        # B, num_head, num_patches, dim_head
        qkv = self.split_heads(self.qkv(x))
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        #If Rotary embeddings are used then apply it in the attention mechanism
        if self.use_rotary:
            assert position is not None, "position required when use_rotary=True"
            #shape is B, num_heads, num_patches, dim_head
            if self.has_cls:
                q, class_token_q = q[:,:,1:,:], q[:,:,:1,:]
                k, class_token_k = k[:,:,1:,:], k[:,:,:1,:]
                q = self.rotary(q, position)
                k = self.rotary(k, position)
                k = torch.concat([class_token_k, k], dim=2)
                q = torch.concat([class_token_q, q], dim=2)
            else:
                q = self.rotary(q, position)
                k = self.rotary(k, position)

        #Compute the attention score
        if self.return_att:
            score = (q@k.transpose(2,3))/(self.dim_head**0.5)

            #If causal we mask all date happening in the future
            if self.is_causal:
                T = score.size(-1)
                mask = torch.triu(torch.ones(T,T, dtype=torch.bool), diagonal=1)
                score = score.masked_fill(mask, float('-inf'))
                
            #Compute the attnetion matrix
            score = score.softmax(dim=-1)
            
            #Apply attention to V
            out = self.dropout(score)@v
        else:
            #Compute the attention score using pytorch built in function
            out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.att_dropout if self.training else 0, is_causal=self.is_causal)
            score = None
        out = out.transpose(1,2)
        out = out.reshape(out.size(0), out.size(1), self.h*self.dim_head)
        out = self.fc(out)
        return self.dropout(out), score


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

       



class TransformerLayer(nn.Module):
    """Define one transformer layer"""
    def __init__(self, embed_dim, nhead = 3, dim_feedforward=2048, dropout=0.1, activation = "gelu", keep_prob = 1, use_rotary = False, has_cls = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=dropout, use_rotary=use_rotary, has_cls = has_cls)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        activations = {"gelu": nn.GELU(), "relu": nn.ReLU()}
        self.activation = activations[activation]
        self.drop_path = DropPath(keep_prob=keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, src, keep):
        #Compute the multihead attention
        z = self.norm1(src)
        attn, _ = self.self_attn(z, position = keep)

        #Apply residual connection and layer normalization
        Z = src + self.drop_path(attn)

        #Apply normalization
        ff = self.norm2(Z)

        #MLP layer
        ff = self.dropout(self.linear2(self.dropout(self.activation(self.linear1(ff)))))

        return (Z + self.drop_path(ff))
    

    
class EncoderDecoder(pl.LightningModule):
    """Basic encoder decoder model following the ViT model"""
    def __init__(self, config = None, use_rotary = True,num_channels = 58, max_embedding = 2000, enc_dim = 768, dec_dim = 256, depth_e = 5, depth_d = 3, mask_prob = 0.7, patch_size = 128):
        super().__init__()
        self.config = config
        self.use_rotary = use_rotary
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim

        #Define the encoder and decoder layers
        self.encoder = nn.ModuleList([TransformerLayer(embed_dim=enc_dim, nhead=8, use_rotary=use_rotary, has_cls=True) for i in range(depth_e)])
        self.decoder = nn.ModuleList([TransformerLayer(embed_dim=dec_dim, nhead=4, use_rotary = use_rotary, has_cls=False) for i in range(depth_d)])

        #Set the probability for a token to be masked
        self.mask_prob = mask_prob

        #Layer convert the encoder dimension to the decoder dimension
        self.encoder_decoder = (nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity())

        self.patch_size = patch_size
        self.patch = PatchEEG(patch_size=patch_size)
        self.mask_token = nn.Parameter(torch.zeros((dec_dim)))
        self.fc = nn.Linear(dec_dim, patch_size)
        self.num_channels = num_channels
        
        #Define the temporal and channel embeddings 
        self.channel_embedding_e = nn.Embedding(num_channels, enc_dim)
        self.channel_embedding_d = nn.Embedding(num_channels, dec_dim)
        self.temporal_embedding_d = TemporalEncoding(embed_dim=dec_dim, max_len=max_embedding)
        self.temporal_embedding_e = TemporalEncoding(embed_dim=enc_dim, max_len=max_embedding)

        self.criterion = nn.MSELoss()
        self.class_token = nn.Parameter(torch.zeros(1,1,enc_dim))
        

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
        

    def masking(self, x):
        """Mask random patch of the eeg input"""

        B, N, C, T = x.shape
        device = x.device

        #Compute total number of patch
        L = N*C

        #Stack all patches in a 1D array
        x_flat = rearrange(x, "b n c t -> b (n c) t")
        
        #Compute the number of patches to keep
        L_keep = int(L * (1-self.mask_prob))

        #Retrieve shuffle as well as ordered indices
        noise = torch.rand(B,L, device=device)
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

    def forward(self, x):
        B, C, T = x.shape
        device = x.device

        #Return the patch eeg with shape (b, n, c, d)
        x = self.patch(x)
        N = x.shape[1]
        L = x.shape[1] * x.shape[2]

        #Define the embeddings
        chan_embedding = self.channel_embedding_e(torch.arange(0,C, device=device))
        chan_embedding = rearrange(chan_embedding, "c d -> 1 1 c d")
        temp_embedding = self.temporal_embedding_e(seq_length = N, num_channel = C)
        temp_embedding = rearrange(temp_embedding, "b (s c) d -> b s c d", c = C)
        
        x += chan_embedding 

        if not self.use_rotary:
            temp_embedding = self.temporal_embedding_e(seq_length = N, num_channel = C)
            temp_embedding = rearrange(temp_embedding, "b (s c) d -> b s c d", c = C)
            x +=  temp_embedding

        #Mask the eeg patches
        x, mask, keep_id = self.masking(x)

        #Concatenate the class token to the eeg
        class_token = self.class_token + self.temporal_embedding_e.get_class_token()
        class_token = class_token.expand(B,1,-1)
        x = torch.cat([class_token, x], dim = 1)
       
        #Pass the sequence through the encoder layers 
        for transformer in self.encoder:
            x = transformer(x, keep_id//C)

        #Retrieve the class token which is not used for decoding
        x, class_token = x[:,1:,:], x[:,:1,:]

        #Get decoder dimensions (B,N*C,D)
        x = self.encoder_decoder(x)
        
        #Get the original ordering of patches
        x = self.to_decoded(x, L, keep_id)
        x = rearrange(x, "b (n c) d -> b n c d", c = C)

        #Get embeddings for decoding
        chan_embedding = self.channel_embedding_d(torch.arange(0,C, device=device))
        chan_embedding = rearrange(chan_embedding, "c d -> 1 1 c d")
        x += chan_embedding

        #Add the temporal embedding if the rotary embedding are not used
        if not self.use_rotary:
            temp_embedding = self.temporal_embedding_d(seq_length = N, num_channel = C)
            temp_embedding = rearrange(temp_embedding, "b (s c) d -> b s c d", c = C)
            x += temp_embedding
        x = rearrange(x, "b n c d -> b (n c) d")

        #Get the eeg sample indices for rotary embeddings
        seq = torch.arange(0,N).repeat_interleave(C)
        seq = seq.expand(B, -1)
        #Pass input through decoder layers
        for transformer in self.decoder:
            x = transformer(x, seq)
        
        #Only keep the initially masked patches
        x = x[torch.where(mask == 1)]
        x = self.fc(x)

        return x, mask
    
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
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mse" 
            }
        }
    
    def training_step(self, batch, batch_idx):
        data, _ = batch
        pred, mask = self(data)
        target, pad = self.patchify_1d(data, self.patch_size)
        target = target.view(target.size(0), -1, self.patch_size)
        
        loss = self.criterion(pred, target[torch.where(mask == 1)].float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, _ = batch
        pred, mask = self(data)
        target, pad = self.patchify_1d(data, self.patch_size)
        target = target.view(target.size(0), -1, self.patch_size)
        mse = self.criterion(pred, target[torch.where(mask == 1)].float())
        mae = F.l1_loss(pred, target[torch.where(mask == 1)].float())
        rmse = torch.sqrt(mse + 1e-8)

        self.log_dict(
        {"val_mse": mse, "val_mae": mae, "val_rmse": rmse},
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
    #data = np.load("data/PCTram/21/trial_60_0.npz")
    #x = data["x"]
    #x = torch.tensor(x, dtype=torch.float32)
    #x = x.unsqueeze(0)
    
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
