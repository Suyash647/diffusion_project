import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)

        q = self.q(h).reshape(B, C, -1)
        k = self.k(h).reshape(B, C, -1)
        v = self.v(h).reshape(B, C, -1)

        attn = torch.bmm(q.permute(0,2,1), k) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0,2,1))
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)