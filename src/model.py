from configs import D, W, H, P, B, N
import torch.nn as nn
import torch

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # 1) create patches and apply affine linear transformations to them
        self.conv = nn.Conv2d(3, D, kernel_size=P, stride=P)

        # 2) positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, N+1, D))

    def forward(self, x):
        # [B, 3, H, W] -> [B, D, H/P, W/P], [B, D, N]
        x = self.conv(x).reshape((B, D, N))

        # [B, D, N] -> [B, N, D]
        x = torch.transpose(x, -2, -1)

        # prepend a classification token to every image's patch sequence
        # [B, N, D] -> [B, N+1, D]
        x = torch.cat((x, nn.Parameter(torch.randn(B, 1, D))), dim=1)

        # add positional embeddings to the patch embeddings
        x = x + self.positional_embeddings

        return x