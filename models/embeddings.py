import torch
import math


def timestep_embedding(timesteps, dim):
    """
    Sinusoidal timestep embeddings (like Transformer positional encoding)
    """
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))

    return emb