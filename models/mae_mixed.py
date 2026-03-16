# Adapted from Vision Transformer (ViT) implementation in MONAI


from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from models.mixedtransformerblock import TransformerBlock
#from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import trunc_normal_
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option
from utilities.embed_utils import get_3d_sincos_pos_embed_from_grid

SUPPORTED_POS_EMBEDDING_TYPES = {"none", "learnable", "sincos"}

__all__ = ["MaskedAutoEncoderViT"]


class MaskedAutoEncoderViT(nn.Module):
    """
    Masked Autoencoder (ViT), based on: "Kaiming et al.,
    Masked Autoencoders Are Scalable Vision Learners <https://arxiv.org/abs/2111.06377>"
    Only a subset of the patches passes through the encoder. The decoder tries to reconstruct
    the masked patches, resulting in improved training speed.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 12,
        masking_ratio_space: float = 0.75,
        masking_ratio_diff: float = 0.5,
        mask_type: int = 2,
        decoder_hidden_size: int = 384,
        decoder_mlp_dim: int = 512,
        decoder_num_layers: int = 4,
        decoder_num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "sincos",
        decoder_pos_embed_type: str = "sincos",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        num_directions: int = 15,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input.
            img_size: dimension of input image.
            patch_size: dimension of patch size
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 512.
            num_layers:  number of transformer blocks. Defaults to 12.
            num_heads: number of attention heads. Defaults to 12.
            masking_ratio: ratio of patches to be masked. Defaults to 0.75.
            decoder_hidden_size: dimension of hidden layer for decoder. Defaults to 384.
            decoder_mlp_dim: dimension of feedforward layer for decoder. Defaults to 512.
            decoder_num_layers: number of transformer blocks for decoder. Defaults to 4.
            decoder_num_heads: number of attention heads for decoder. Defaults to 12.
            proj_type: position embedding layer type. Defaults to "conv".
            pos_embed_type: position embedding layer type. Defaults to "sincos".
            decoder_pos_embed_type: position embedding layer type for decoder. Defaults to "sincos".
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dimensions. Defaults to 3.
            qkv_bias: apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.
        Examples::
            # for single channel input with image size of (96,96,96), and sin-cos positional encoding
            >>> net = MaskedAutoEncoderViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16),
            pos_embed_type='sincos')
            # for 3-channel with image size of (128,128,128) and a learnable positional encoding
            >>> net = MaskedAutoEncoderViT(in_channels=3, img_size=128, patch_size=16, pos_embed_type='learnable')
            # for 3-channel with image size of (224,224) and a masking ratio of 0.25
            >>> net = MaskedAutoEncoderViT(in_channels=3, img_size=(224,224), patch_size=(16,16), masking_ratio=0.25,
            spatial_dims=2)
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate should be between 0 and 1, got {dropout_rate}.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        if decoder_hidden_size % decoder_num_heads != 0:
            raise ValueError("decoder_hidden_size should be divisible by decoder_num_heads.")

        self.patch_size = patch_size
        self.img_size = img_size
        self.spatial_dims = spatial_dims
        self.num_directions = num_directions
        for m, p in zip(self.img_size, self.patch_size):
            if m % p != 0:
                raise ValueError(f"patch_size={patch_size} should be divisible by img_size={img_size}.")
        self.S = (self.img_size[0]//self.patch_size[0])*(self.img_size[1]//self.patch_size[1])*(self.img_size[2]//self.patch_size[2])
        self.decoder_hidden_size = decoder_hidden_size

        self.mask_type = mask_type
        if masking_ratio_space <= 0 or masking_ratio_space >= 1:
            raise ValueError(f"masking_ratio should be in the range (0, 1), got {masking_ratio_space}.")
        if masking_ratio_diff <= 0 or masking_ratio_diff >= 1:
            raise ValueError(f"masking_ratio should be in the range (0, 1), got {masking_ratio_diff}.")
        self.mask_ratio_space = masking_ratio_space
        self.mask_ratio_diff = masking_ratio_diff

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )

        self.patch_mlp = nn.Linear(self.patch_size[0]*self.patch_size[1]*self.patch_size[2], hidden_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches, hidden_size//2))
        self.qpos_embed = nn.Linear(3,hidden_size//2)

        blocks = [
            TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
            for _ in range(num_layers)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(hidden_size)
        # decoder
        self.decoder_embed = nn.Linear(hidden_size, decoder_hidden_size)

        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

        self.decoder_pos_embed_type = look_up_option(decoder_pos_embed_type, SUPPORTED_POS_EMBEDDING_TYPES)
        self.decoder_pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches, decoder_hidden_size//2))

        decoder_blocks = [
            TransformerBlock(decoder_hidden_size, decoder_mlp_dim, decoder_num_heads, dropout_rate, qkv_bias, save_attn)
            for _ in range(decoder_num_layers)
        ]
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.ln2 = nn.LayerNorm(decoder_hidden_size)
        self.decoder_pred = nn.Sequential(nn.Linear(decoder_hidden_size, int(np.prod(self.patch_size)) * in_channels))
        self.mask_to_apply = 0 # control which type of mask for alternate
        self._init_weights()

    def _init_weights(self):
        """
        similar to monai/networks/blocks/patchembedding.py for the decoder positional encoding and for mask and
        classification tokens
        """
        if self.decoder_pos_embed_type == "none":
            pass
        elif self.decoder_pos_embed_type == "learnable":
            trunc_normal_(self.decoder_pos_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0)
            trunc_normal_(self.pos_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.decoder_pos_embed_type == "sincos":
            grid_size = []
            for in_size, pa_size in zip(self.img_size, self.patch_size):
                grid_size.append(in_size // pa_size)
            with torch.no_grad():
                self.decoder_pos_embedding = build_sincos_position_embedding(
                    grid_size, self.decoder_hidden_size//2, self.spatial_dims
                )
        else:
            raise ValueError(f"decoder_pos_embed_type {self.decoder_pos_embed_type} not supported.")

        # initialize patch_embedding like nn.Linear (instead of nn.Conv2d)
        trunc_normal_(self.mask_tokens, mean=0.0, std=0.02, a=-2.0, b=2.0)
        trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def _masking(self, x, masking_ratio: float | None = None):
        """Apply spatial masking by randomly selecting tokens to keep."""
        batch_size, num_tokens, _ = x.shape
        percentage_to_keep = 1 - masking_ratio if masking_ratio is not None else 1 - self.masking_ratio
        selected_indices = torch.multinomial(
            torch.ones(batch_size, num_tokens), int(percentage_to_keep * num_tokens), replacement=False
        )
        x_masked = x[torch.arange(batch_size).unsqueeze(1), selected_indices]  # gather the selected tokens
        mask = torch.ones(batch_size, num_tokens, dtype=torch.int).to(x.device)
        mask[torch.arange(batch_size).unsqueeze(-1), selected_indices] = 0

        return x_masked, selected_indices, mask

    def _diff_masking(self, x, masking_ratio: float | None = None):
        """Apply diffusion masking across directions by randomly selecting tokens to keep."""
        batch_size, num_tokens, s, _ = x.shape
        percentage_to_keep = 1 - masking_ratio if masking_ratio is not None else 1 - self.masking_ratio
        selected_indices = torch.multinomial(
            torch.ones(batch_size, num_tokens), int(percentage_to_keep * num_tokens), replacement=False
        )
        x_masked = x[torch.arange(batch_size).unsqueeze(1), selected_indices]  # gather the selected tokens
        mask = torch.ones(batch_size, num_tokens, s, dtype=torch.int).to(x.device)
        mask[torch.arange(batch_size).unsqueeze(-1), selected_indices] = 0

        return x_masked, selected_indices, mask

    def get_positional_embed(self,positions):
        """Generate 3D sinusoidal positional embeddings from patch centers."""
        P = self.patch_size
        center_offset = [p//2 for p in P]
        all_embs = []
        with torch.no_grad():
            imspace_center = positions[:, center_offset[0]::P[0], center_offset[1]::P[1], center_offset[2]::P[2],:]
            imspace_center = rearrange(imspace_center,'b h w z c -> b (h w z) c')
            b,v,c = imspace_center.shape
            imspace_center = rearrange(imspace_center,'b v c -> (b v) c')
            ee = get_3d_sincos_pos_embed_from_grid(self.decoder_hidden_size//2, imspace_center[:,1], imspace_center[:,0], imspace_center[:,2])
            ee = rearrange(ee,'(b v) c -> b v c', v=v)
            self.decoder_pos_embedding = nn.Parameter(ee,requires_grad=False)
        return self.decoder_pos_embedding

    def full_positional_embed(self, positions,spherical):
        """Combine spatial and directional positional embeddings into full embedding."""
        time_embed = self.qpos_embed(spherical).unsqueeze(2)
        t = time_embed.shape[1]
        space_embed = self.get_positional_embed(positions).unsqueeze(1)
        s = space_embed.shape[2]
        time_embed = time_embed.expand(-1, -1, s, -1)
        # Expand B to shape [b, s, t, d]
        space_embed = space_embed.expand(-1, t, -1, -1)
        # Concatenate along the last dimension to get [b, s, t, 2*d]
        full_pos_embed = torch.cat([space_embed,time_embed],dim=-1) 
        return full_pos_embed

    def forward_patch_embed(self,x,spherical, positions):
        """Convert input to patches and add positional embeddings."""
        x = self.patchify(x)
        x = self.patch_mlp(x)
        b, t, s, d = x.shape
        full_pos_embed = self.full_positional_embed(positions,spherical)
        x = x + full_pos_embed
        x = rearrange(x,'b t s d -> (b t) s d')
        return x, b, t, s, d

    def forward_mask(self,x,t,seed = None):
        """Apply masking strategy (spatial, diffusion, or alternating) to input tokens."""
        if seed:
            torch.manual_seed(seed)

        if self.mask_type<=1:
            self.mask_to_apply = self.mask_type
        elif self.mask_type==2: #alternating
            self.mask_to_apply = 1 - self.mask_to_apply
        if self.mask_to_apply == 1: #diffusion masking
            x = rearrange(x,'(b t) s d -> b t s d', t=t)
            x, selected_indices, mask = self._diff_masking(x, masking_ratio=self.mask_ratio_diff)
        elif self.mask_to_apply == 0: #space masking
            x, selected_indices, mask = self._masking(x, masking_ratio=self.mask_ratio_space)
            x = rearrange(x,'(b t) s d -> b t s d', t=t)
        B, T, S, D = x.shape
        return x, selected_indices, mask, B, T, S ,D 

    def forward_encode(self,x,b,t,s):
        """Pass tokens through encoder transformer blocks."""
        #CLS
        x = rearrange(x,'b t s d -> b (s t) d')
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # encoder transformer blocks
        for blk in self.blocks:
            x = blk(x, b,t,s)
        x = self.ln(x)
        return x

    def forward_unmask(self, x, selected_indices, mask, s, t, T):
        """Restore full sequence by inserting mask tokens for masked positions."""
        if self.mask_to_apply == 1: #diffusion masking
            x = rearrange(x,'b (s t) d -> b t s d', t=T)
            x_ = self.mask_tokens.repeat(x.shape[0], mask.shape[1], mask.shape[2], 1)
            x_[torch.arange(x.shape[0]).unsqueeze(-1), selected_indices] = x
        elif self.mask_to_apply == 0: #space masking
            x = rearrange(x,'b (s t) d -> (b t) s d', t=t)
            x_ = self.mask_tokens.repeat(x.shape[0], mask.shape[1], 1)
            x_[torch.arange(x.shape[0]).unsqueeze(-1), selected_indices] = x
            x_ = rearrange(x_,'(b t) s d -> b t s d',s=s,t=t)
            mask = rearrange(mask,'(b t) s -> b t s', t=t)
        return x_, mask

    def forward_decode(self, x_, spherical, positions, cls_token, b, t, s):
        """Pass unmasked tokens through decoder transformer blocks to reconstruct input."""
        full_pos_embed = self.full_positional_embed(positions,spherical)
        x_ = x_ + full_pos_embed
        x_ = rearrange(x_,'b t s d -> (b t) s d')      
        x_ = rearrange(x_,'(b t) s d -> b (s t) d',t=t)
        # add cls token back
        x = torch.cat([cls_token, x_], dim=1)
        for blk in self.decoder_blocks:
            x = blk(x, b,t,s)
        x = self.ln2(x)
        x = self.decoder_pred(x)

        return x

    def forward_nomask(self,x,spherical, positions):
        """Forward pass without masking - used for inference/embedding extraction."""
        # Patch and positional embedding
        x, b, t, s, d = self.forward_patch_embed(x,spherical, positions)
        x = rearrange(x,'(b t) s d -> b t s d', t=t)
        x = self.forward_encode(x,b,t,s)
        cls_token = x[:,0,:].unsqueeze(1)
        x = x[:, 1:, :] # no cls token
        x = rearrange(x,' b (s t) d -> b t s d', t=t)
        return x, cls_token


    def forward(self, x, spherical, positions, seed = None):
        """Main forward pass: patchify, mask, encode, decode, and reconstruct."""
        # Patch and positional embedding
        x, b, t, s, d = self.forward_patch_embed(x,spherical, positions)

        # Masking
        x, selected_indices, mask, B, T, S ,D = self.forward_mask(x,t, seed)
        
        # Encoder
        x = self.forward_encode(x,b,T,S)

        # mid layer
        x = self.decoder_embed(x)
        cls_token = x[:,0,:].unsqueeze(1)
        x = x[:, 1:, :] # no cls token

        # unmasking
        x_, mask = self.forward_unmask(x,selected_indices,mask,s,t,T)

        x = self.forward_decode(x_,spherical,positions, cls_token,b,t,s)

        cls = x[:, 0, :]
        x = x[:,1:,:]
        x = rearrange(x,'b (s t) d -> b t s d', t=t)
       

        return x, mask

    def patchify(self, imgs):
        """Convert 3D+T images into patches.
        imgs: (N, H, W, Z, T)
        x: (N, T, S, patch_size**2)
        """
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        p3 = self.patch_size[2]
        t = self.num_directions
        h = imgs.shape[1] // p1
        w = imgs.shape[2] // p2
        z = imgs.shape[3] // p3
        x = rearrange(imgs, "b (h p1) (w p2) (z p3) t -> b t (h w z) (p1 p2 p3)", h=h,p1=p1,w=w,z=z,p2=p2,p3=p3)
        return x # b x t x s x p
    
    def unpatchify(self, x):
        """Convert patches back to 3D+T images.
        x: (N, T, S, patch_size**3)
        imgs: (N, H, W, Z, T)
        """
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        p3 = self.patch_size[2]
        t = self.num_directions
        h = self.img_size[-3] // p1
        w = self.img_size[-2] // p2
        z = self.img_size[-1] // p3
        rec = rearrange(x, "b t (h w z) (p1 p2 p3) -> b (h p1) (w p2) (z p3) t", h=h,p1=p1,w=w,z=z,p2=p2,p3=p3)

        return rec