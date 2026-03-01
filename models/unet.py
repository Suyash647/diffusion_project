import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embeddings import timestep_embedding


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)

        q = self.q(x_norm).reshape(b, c, -1)
        k = self.k(x_norm).reshape(b, c, -1)
        v = self.v(x_norm).reshape(b, c, -1)

        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (c ** 0.5), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)

        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h += self.time_mlp(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, base=64, time_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Down
        self.conv0 = nn.Conv2d(3, base, 3, padding=1)
        self.res1 = ResBlock(base, base, time_dim)
        self.res2 = ResBlock(base, base*2, time_dim)
        self.res3 = ResBlock(base*2, base*4, time_dim)
        self.res4 = ResBlock(base*4, base*8, time_dim)

        self.attn = SelfAttention(base*2)

        self.pool = nn.AvgPool2d(2)

        # Middle
        self.mid1 = ResBlock(base*8, base*8, time_dim)
        self.mid2 = ResBlock(base*8, base*8, time_dim)

        # Up
        self.up4 = ResBlock(base*16, base*4, time_dim)
        self.up3 = ResBlock(base*8, base*2, time_dim)
        self.up2 = ResBlock(base*4, base, time_dim)
        self.up1 = ResBlock(base*2, base, time_dim)

        self.final = nn.Conv2d(base, 3, 1)

    def forward(self, x, t):
        t = timestep_embedding(t, 256)
        t = self.time_mlp(t)

        x1 = self.res1(self.conv0(x), t)
        x2 = self.res2(self.pool(x1), t)
        x2 = self.attn(x2)
        x3 = self.res3(self.pool(x2), t)
        x4 = self.res4(self.pool(x3), t)

        m = self.mid1(self.pool(x4), t)
        m = self.mid2(m, t)

        u4 = F.interpolate(m, scale_factor=2)
        u4 = self.up4(torch.cat([u4, x4], dim=1), t)

        u3 = F.interpolate(u4, scale_factor=2)
        u3 = self.up3(torch.cat([u3, x3], dim=1), t)

        u2 = F.interpolate(u3, scale_factor=2)
        u2 = self.up2(torch.cat([u2, x2], dim=1), t)

        u1 = F.interpolate(u2, scale_factor=2)
        u1 = self.up1(torch.cat([u1, x1], dim=1), t)

        return self.final(u1)