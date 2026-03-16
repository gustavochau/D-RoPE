import torch
#import numpy as np

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid_x, grid_y, grid_z):
    if embed_dim % 6 != 0:
        raise ValueError("embed_dim must be a multiple of 6")

    # Each direction gets 1/3 of the total dimensions
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid_y)  # (M, D/3)
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid_x)  # (M, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid_z)  # (M, D/3)

    emb = torch.cat([emb_y, emb_x, emb_z], dim=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a 1D torch tensor of positions to encode, shape (M,)
    returns: torch tensor of shape (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    device = pos.device
    dtype = pos.dtype

    omega = torch.arange(embed_dim // 2, dtype=dtype, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


# def get_3d_sincos_pos_embed_from_grid(embed_dim, grid_x,grid_y,grid_z):
#     if embed_dim % 6 != 0:
#         raise ValueError("embed_dim must be a multiple of 6")

#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid_y)  # (H*W*Z, D/3)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid_x)  # (H*W*Z, D/3)
#     emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid_z)  # (H*W*Z, D/3)
#     print('emb_h')
#     print(emb_h.shape)
#     emb = np.concatenate([emb_h, emb_w, emb_z], axis=1)  # (H*W, D)
#     return emb#torch.from_numpy(emb,type=torch.float32)


# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     """
#     embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
#     """
#     if embed_dim % 2 != 0:
#         raise ValueError("embed_dim must be even")

#     omega = np.arange(embed_dim // 2, dtype=float)
#     omega /= embed_dim / 2.0
#     omega = 1.0 / 10000**omega  # (D/2,)

#     pos = pos.reshape(-1)  # (M,)
#     out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

#     emb_sin = np.sin(out)  # (M, D/2)
#     emb_cos = np.cos(out)  # (M, D/2)

#     emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
#     return emb