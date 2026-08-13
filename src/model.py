from transformer_layers import TransformerEncoder
from configs import D, W, H, P, B, N, dropout_P
import torch.nn as nn
import torch

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # 1) create patches and apply affine linear transformations to them
        self.conv = nn.Conv2d(3, D, kernel_size=P, stride=P)

        # 2) positional embeddings + dropout
        self.positional_embeddings = nn.Parameter(torch.randn(1, N+1, D))
        self.dropout = nn.Dropout(dropout_P)

        # 3) transformer encoder
        self.transformer_encoder = TransformerEncoder()

        # 4) layer normalize encoded classification tokens
        self.layer_norm = nn.LayerNorm(D)

    def forward(self, x):
        # [B, 3, H, W] -> [B, D, H/P, W/P], [B, D, N]
        x = self.conv(x).reshape((B, D, N))

        # [B, D, N] -> [B, N, D]
        x = torch.transpose(x, -2, -1)

        # prepend a classification token to every image's patch sequence
        # [B, N, D] -> [B, N+1, D]
        x = torch.cat((x, nn.Parameter(torch.randn(B, 1, D))), dim=1)

        # add positional embeddings to the patch embeddings
        x = self.dropout(x + self.positional_embeddings)

        # feed patch embeddings to the transformer encoder
        x = self.transformer_encoder(x)

        # extract encoded classification tokens
        x = torch.select(x, -2, 0).reshape(B, 1, D)

        return x