# Adapted from Vision Transformer (ViT) implementation in MONAI

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from monai.utils import optional_import
from einops import repeat, rearrange
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class SABlock_rope(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def swap_and_negate_even_pairs(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation [-x2, x1, -x4, x3, ..., -xn, xn-1] 
        along the last dimension of a (b, h, d) tensor, assuming d is even.
        
        Parameters:
        batch (torch.Tensor): Input tensor of shape (b, h, d), where d is even.
        
        Returns:
        torch.Tensor: Transformed tensor of the same shape.
        """
        b, l, h, d = batch.shape
        assert d % 2 == 0, "The last dimension must be even."
        
        reshaped = batch.view(b, l, h, d // 2, 2)  # shape: (b, h, d/2, 2)
        swapped = torch.stack([-reshaped[..., 1], reshaped[..., 0]], dim=-1)
        return swapped.view(b, l, h, d)

    def weight_k(self, q, k, comp):
        """
        Apply D-RoPE weights to the key tensor.
        
        Args:
            q (torch.Tensor): Query tensor of shape (b, l, h_q, d).
            k (torch.Tensor): Key tensor of shape (b, l, h_k, d).
            comp (torch.Tensor): Position embedding components of shape (h_q, h_k, d_reduced).
        
        Returns:
            torch.Tensor: Weighted key tensor with rotary embeddings applied, shape (b, l, h_q, h_k, d).
        """
        b, l, h_q, d = q.shape
        h_k = k.shape[2]
        
        Wk = torch.empty((b, l, h_q, h_k, d), device=q.device, dtype=q.dtype)
        swapped_k = self.swap_and_negate_even_pairs(k)
        swapped_k = repeat(swapped_k,'b l hk d -> b l hq hk d',hq=h_q)
        k = repeat(k,'b l hk d -> b l hq hk d',hq=h_q)
        comp = rearrange(comp, 'b hq hk d -> b 1 hq hk d')
        Wk = torch.cos(comp)*k + torch.sin(comp)*swapped_k
        return Wk.to(torch.float32)

    def forward(self, x, components):
        """
        Forward pass of the self-attention block with D-RoPE.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, h, hidden_size).
            components (torch.Tensor): Position embedding components for rotary embeddings.
        
        Returns:
            torch.Tensor: Output tensor after self-attention, same shape as input.
        """
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        Wk = self.weight_k(q, k, components)
        att_mat = (torch.einsum("blhd,blhyd->blhy", q, Wk)*self.scale).softmax(dim=-1)
        
        if self.save_attn:
            # Store attention matrix without gradients for visualization/analysis
            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x
