import torch
import torch.nn as nn
import math


# ------------------------------------------------
# Sinusoidal Time Embedding
# ------------------------------------------------

def timestep_embedding(timesteps, dim):

    device = timesteps.device
    half_dim = dim // 2

    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    return emb


# ------------------------------------------------
# Self Attention
# ------------------------------------------------

class SelfAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)

        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):

        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(B, C, H * W)

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        scale = C ** -0.5

        attn = torch.softmax(
            torch.bmm(q.transpose(1, 2), k) * scale,
            dim=-1
        )

        out = torch.bmm(v, attn.transpose(1, 2))
        out = self.proj(out)

        out = out.view(B, C, H, W)

        return out + residual


# ------------------------------------------------
# Residual Block
# ------------------------------------------------

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


# ------------------------------------------------
# Downsample
# ------------------------------------------------

class Downsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ------------------------------------------------
# Upsample
# ------------------------------------------------

class Upsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):

        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ------------------------------------------------
# Diffusion UNet
# ------------------------------------------------

class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        time_dim=256
    ):
        super().__init__()

        self.time_dim = time_dim

        # time embedding

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # initial conv

        self.conv0 = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # -------------------
        # Down path
        # -------------------

        self.down1 = Block(base_channels, base_channels * 2, time_dim)
        self.attn1 = SelfAttention(base_channels * 2)

        self.downsample1 = Downsample(base_channels * 2)

        self.down2 = Block(base_channels * 2, base_channels * 4, time_dim)
        self.attn2 = SelfAttention(base_channels * 4)

        self.downsample2 = Downsample(base_channels * 4)

        # -------------------
        # Bottleneck
        # -------------------

        self.bot1 = Block(base_channels * 4, base_channels * 8, time_dim)
        self.attn_mid = SelfAttention(base_channels * 8)
        self.bot2 = Block(base_channels * 8, base_channels * 4, time_dim)

        # -------------------
        # Up path
        # -------------------

        self.up1 = Upsample(base_channels * 4)

        self.up_block1 = Block(
            base_channels * 8,
            base_channels * 2,
            time_dim
        )

        self.up2 = Upsample(base_channels * 2)

        self.up_block2 = Block(
            base_channels * 4,
            base_channels,
            time_dim
        )

        # final conv

        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):

        # time embedding
        t = timestep_embedding(t, self.time_dim)
        t = self.time_mlp(t)

        # initial
        x0 = self.conv0(x)

        # down
        x1 = self.down1(x0, t)
        x1 = self.attn1(x1)

        x2 = self.downsample1(x1)

        x2 = self.down2(x2, t)
        x2 = self.attn2(x2)

        x3 = self.downsample2(x2)

        # bottleneck
        x3 = self.bot1(x3, t)
        x3 = self.attn_mid(x3)
        x3 = self.bot2(x3, t)

        # up
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up_block1(x, t)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_block2(x, t)

        return self.final(x)