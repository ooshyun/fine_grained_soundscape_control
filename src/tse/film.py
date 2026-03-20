import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, input_channels, embedding_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.a = nn.Linear(embedding_channels, input_channels)
        self.b = nn.Linear(embedding_channels, input_channels)

    def forward(self, x, emb):
        # x: [B, C, *]
        # y: [B, D]
        a = self.a(emb)
        b = self.b(emb)
        while len(a.shape) != len(x.shape):
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)

        return x * a + b
