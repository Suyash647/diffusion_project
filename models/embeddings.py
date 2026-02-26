import torch
import math


def timestep_embedding(t, dim):
    """
    Create sinusoidal timestep embeddings.
    t: tensor of shape [batch]
    returns: tensor of shape [batch, dim]
    """
    device = t.device
    half_dim = dim // 2

    emb = torch.exp(
        torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1))
    )

    emb = t[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    return emb