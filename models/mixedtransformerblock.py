# Adapted from Vision Transformer (ViT) implementation in MONAI and https://github.com/facebookresearch/TimeSformer

from __future__ import annotations
import torch
import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock
from einops import rearrange

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp_diffusion = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1_diffusion = nn.LayerNorm(hidden_size)
        self.attn_diffusion = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2_diffusion = nn.LayerNorm(hidden_size)

        self.mlp_space = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1_space = nn.LayerNorm(hidden_size)
        self.attn_space = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2_space = nn.LayerNorm(hidden_size)

    def diffusion_forward(self, x):
        """
        Forward pass for diffusion space attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b*s, t, d) where t is the number of diffusion steps.
        
        Returns:
            torch.Tensor: Output tensor after diffusion space attention and MLP, same shape as input.
        """
        x = x + self.attn_diffusion(self.norm1_diffusion(x))
        x = x + self.mlp_diffusion(self.norm2_diffusion(x))
        return x

    def space_forward(self, x):
        """
        Forward pass for spatial attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b*t, s, d).
        
        Returns:
            torch.Tensor: Output tensor after spatial attention and MLP, same shape as input.
        """
        x = x + self.attn_space(self.norm1_space(x))
        x = x + self.mlp_space(self.norm2_space(x))
        return x

    def forward(self, x, b, t, s):
        """
        Forward pass applying mixed spatial-diffusion attention.
        
        Processes input through diffusion space attention followed by spatial attention,
        handling CLS token separately.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, s*t+1, d) where the first token is CLS.
            b (int): Batch size.
            t (int): Number of diffusion directions.
            s (int): Number of spatial tokens per diffusion step.
        
        Returns:
            torch.Tensor: Output tensor after mixed attention, shape (b, s*t+1, d).
        """
        # Extract cls token
        init_cls_token = x[:,0,:].unsqueeze(1)
        x = x[:,1:,:]

        # Attention in diffusion space
        x = rearrange(x,'b (s t) d -> (b s) t d', t=t)
        x_diffusion = self.diffusion_forward(x)
        x_diffusion = rearrange(x_diffusion,'(b s) t d -> b (s t) d', b=b, t=t)

        # Prepare for spatial attention
        cls_token = init_cls_token.repeat(1, t, 1)
        cls_token = rearrange(cls_token, 'b t d -> (b t) d').unsqueeze(1)

        xs = x_diffusion
        xs = rearrange(xs, 'b (s t) m -> (b t) s m',b=b, t=t)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.attn_space(self.norm1_space(xs))

        # Process CLS token by averaging across diffusion steps
        cls_token = res_spatial[:,0,:]
        cls_token = rearrange(cls_token, '(b t) m -> b t m',b=b,t=t)
        cls_token = torch.mean(cls_token,1,True)
        res_spatial = res_spatial[:,1:,:]
        res_spatial = rearrange(res_spatial, '(b t) s m -> b (s t) m',b=b, t=t)
        res = res_spatial
        x = x_diffusion

        # Apply MLP with residual connections
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.mlp_space(self.norm2_space(x))

        return x
