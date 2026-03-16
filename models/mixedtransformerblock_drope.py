# Adapted from Vision Transformer (ViT) implementation in MONAI and https://github.com/facebookresearch/TimeSformer


from __future__ import annotations
import torch
import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock
from models.selfattention_drope import SABlock_rope
from einops import rearrange
import math
import torch.nn.functional as F

class FlashSelfAttention(nn.Module):
    """
    Lightweight flash attention wrapper using PyTorch's
    `scaled_dot_product_attention` for the core compute.

    Notes:
    - When `save_attn=True` we compute attention weights with a
      conventional (non-flash) softmax over QK^T so they can be
      inspected. That path does not use the flash kernel and may
      be more memory/time consuming.
    - Input/Output shapes follow (batch, seq, hidden_size).
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False, save_attn: bool = False) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.save_attn = save_attn
        # will hold the last attention weights if requested
        self.attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using flash attention for efficient scaled dot-product attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq, hidden).
        
        Returns:
            torch.Tensor: Output tensor after attention, same shape as input.
        """
        b, s, d = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use PyTorch's scaled_dot_product_attention which will use flash kernels when available
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(b, s, d)
        out = self.out_proj(attn_out)
        out = self.dropout(out)

        if self.save_attn:
            # Compute conventional attention weights for inspection (non-flash fallback)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            # Store either full weights or their mean across heads to save memory
            try:
                self.attn = attn_weights.detach()
            except Exception:
                self.attn = attn_weights.detach().mean(1)

        return out

class TransformerBlock(nn.Module):

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
        self.attn_diffusion = SABlock_rope(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2_diffusion = nn.LayerNorm(hidden_size)

        self.mlp_space = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1_space = nn.LayerNorm(hidden_size)
        # Use flash-backed scaled-dot-product attention for spatial attention
        self.attn_space = FlashSelfAttention(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2_space = nn.LayerNorm(hidden_size)

    def diffusion_forward(self, x, components):
        """
        Forward pass for diffusion space attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b*s, t, d) where t is the number of diffusion steps.
            components (torch.Tensor): Position embedding components for rotary embeddings.
        
        Returns:
            torch.Tensor: Output tensor after diffusion space attention and MLP, same shape as input.
        """
        x = x + self.attn_diffusion(self.norm1_diffusion(x), components)
        x = x + self.mlp_diffusion(self.norm2_diffusion(x))
        return x

    def space_forward(self, x):
        """
        Forward pass for spatial attention using flash attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b*t, s, d).
        
        Returns:
            torch.Tensor: Output tensor after spatial attention and MLP, same shape as input.
        """
        x = x + self.attn_space(self.norm1_space(x))
        x = x + self.mlp_space(self.norm2_space(x))
        return x

    def forward(self, x, b, t, s, components):
        """
        Forward pass applying mixed spatial-diffusion attention.
        
        Processes input through diffusion space attention followed by
        spatial attention (with flash attention), handling CLS token separately.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, s*t+1, hidden_dim) where the first token is CLS.
            b (int): Batch size.
            t (int): Number of diffusion directions.
            s (int): Number of spatial tokens per diffusion step.
            components (torch.Tensor): Position embedding components for diffusion space rotary embeddings.
        
        Returns:
            torch.Tensor: Output tensor after mixed attention, shape (b, s*t+1, hidden_dim).
        """
        # Extract cls token
        init_cls_token = x[:,0,:].unsqueeze(1)
        x = x[:,1:,:]

        # Attention in diffusion space
        x = rearrange(x,'b (s t) d -> (b s) t d', t=t)
        x_diffusion = self.diffusion_forward(x, components)
        x_diffusion = rearrange(x_diffusion,'(b s) t d -> b (s t) d', b=b, t=t)

        # Prepare for spatial attention
        cls_token = init_cls_token.repeat(1, t, 1)
        cls_token = rearrange(cls_token, 'b t d -> (b t) d').unsqueeze(1)

        xs = x_diffusion
        xs = rearrange(xs, 'b (s t) m -> (b t) s m', b=b, t=t)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.attn_space(self.norm1_space(xs))

        # Process CLS token by averaging across diffusion steps
        cls_token = res_spatial[:,0,:]
        cls_token = rearrange(cls_token, '(b t) m -> b t m', b=b, t=t)
        cls_token = torch.mean(cls_token, 1, True)
        res_spatial = res_spatial[:,1:,:]
        res_spatial = rearrange(res_spatial, '(b t) s m -> b (s t) m', b=b, t=t)
        res = res_spatial
        x = x_diffusion

        # Apply MLP with residual connections
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.mlp_space(self.norm2_space(x))

        return x
