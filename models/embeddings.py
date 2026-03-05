import torch
import math


def timestep_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2

    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    return emb