import torch
import torch.nn as nn
from models.embeddings import timestep_embedding


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, t):

        # timestep embedding
        t_emb = timestep_embedding(t, 32)
        t_emb = self.time_mlp(t_emb)

        # reshape for adding to image
        t_emb = t_emb[:, :, None, None]

        x = self.relu(self.conv1(x) + t_emb)
        x = self.relu(self.conv2(x) + t_emb)
        x = self.relu(self.conv3(x) + t_emb)
        x = self.conv4(x)

        return x