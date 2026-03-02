import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ----------------------------
# Sinusoidal Time Embedding
# ----------------------------
def timestep_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


# ----------------------------
# Self Attention
# ----------------------------
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x

        x = self.norm(x)
        x = x.view(B, C, H * W)

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(C), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))

        out = self.proj(out)
        out = out.view(B, C, H, W)

        return out + x_in


# ----------------------------
# Residual Block
# ----------------------------
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU()

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.res_conv(x)


# ----------------------------
# Diffusion UNet
# ----------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_dim=256):
        super().__init__()

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial conv
        self.conv0 = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down blocks
        self.down1 = Block(base_channels, base_channels * 2, time_dim)
        self.down2 = Block(base_channels * 2, base_channels * 4, time_dim)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = Block(base_channels * 4, base_channels * 4, time_dim)
        self.attn = SelfAttention(base_channels * 4)
        self.bot2 = Block(base_channels * 4, base_channels * 4, time_dim)

        # Up blocks
        
        self.up1 = Block(base_channels * 8,
                 base_channels * 2, time_dim)
        self.up2 = Block(base_channels * 4,
                 base_channels, time_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.final = nn.Conv2d(base_channels, out_channels, 1)

        self.time_dim = time_dim

    def forward(self, x, t):
        # Time embedding
        t = timestep_embedding(t, self.time_dim)
        t = self.time_mlp(t)

        # Down
        x0 = self.conv0(x)
        x1 = self.down1(x0, t)
        x2 = self.pool(x1)
        x2 = self.down2(x2, t)
        x3 = self.pool(x2)

        # Bottleneck
        x3 = self.bot1(x3, t)
        x3 = self.attn(x3)
        x3 = self.bot2(x3, t)

        # Up
        x4 = self.upsample(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.up1(x4, t)

        x5 = self.upsample(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.up2(x5, t)

        return self.final(x5)