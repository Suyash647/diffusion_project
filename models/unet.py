import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embeddings import timestep_embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(t_emb)
        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Down
        self.conv0 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim)

        self.pool = nn.AvgPool2d(2)

        # Middle
        self.mid1 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.mid2 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        # Up
        self.up3 = ResidualBlock(base_channels * 16, base_channels * 4, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.up1 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)

        self.final = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, 256)
        t_emb = self.time_mlp(t_emb)

        # Down
        x1 = self.conv0(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(self.pool(x2), t_emb)
        x4 = self.down3(self.pool(x3), t_emb)

        # Middle
        xm = self.mid1(self.pool(x4), t_emb)
        xm = self.mid2(xm, t_emb)

        # Up
        u3 = F.interpolate(xm, scale_factor=2, mode="nearest")
        u3 = torch.cat([u3, x4], dim=1)
        u3 = self.up3(u3, t_emb)

        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.up2(u2, t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up1(u1, t_emb)

        return self.final(u1)