from configs import D, P, B, N
import torch.nn as nn
import torch

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # 1) create patches and apply affine linear transformations to them
        self.conv = nn.Conv2d(3, D, kernel_size=P, stride=P)

        # 2) positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, N, D))

    def forward(self, x):
        # [B, 3, H, W] -> [B, D, P, P], [B, D, P^2]
        x = self.conv(x).reshape((B, D, P*P))

        # [B, D, P^2] -> [B, P^2, D]
        x = torch.transpose(x, 1, 2)

        # prepend a classification token to every image's patch sequence
        x = torch.cat((x, nn.Parameter(torch.randn(B, 1, D))), dim=1)

        # add positional embeddings to the patch embeddings
        x = x + self.positional_embeddings

        return x

model = VisionTransformer()

inputs = torch.randn(B, 3, P, P)
outputs = model(inputs)

print(outputs.shape)








