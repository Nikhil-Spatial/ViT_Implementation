from configs import D, P, B
import torch.nn as nn
import torch

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # 1) create patches and apply affine linear transformations to them
        self.conv = nn.Conv2d(3, D, kernel_size=P, stride=P)

    def forward(self, x):
        # [B, D, P, P] -> [B, D, P^2]
        x = self.conv(x).reshape((B, D, P*P))

        # [B, D, P^2] -> [B, P^2, D]
        x = torch.transpose(x, 1, 2)

        # prepend a classification token to every image's patch sequence
        x = torch.cat((x, torch.randn(B, 1, D)), dim=1)










