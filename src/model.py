from src.configs import D, W, H, P, N, dropout_P, CLASSES, device
from src.transformer_layers import TransformerEncoder
import torch.nn.functional as F
import torch.nn as nn
import torch

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # create patches and apply affine linear transformations to them
        self.conv_layer = nn.Conv2d(3, D, kernel_size=P, stride=P)

        # classification token
        self.classification_token = nn.Parameter(torch.randn(1, 1, 512))

        # positional embeddings + dropout
        self.positional_embeddings = nn.Parameter(torch.randn(1, N+1, D))
        self.dropout = nn.Dropout(dropout_P)

        # transformer encoder
        self.transformer_encoder = TransformerEncoder()

        # layer normalize encoded classification tokens
        self.layer_norm = nn.LayerNorm(D)

        # MLP head - one hidden layer and an output layer that projects the
        # classification token's embedded dimensions to the number of classes
        self.hidden_layer = nn.Linear(D, D)
        self.output_layer = nn.Linear(D, CLASSES)

    def forward(self, x):
        # batch size (the last batch received from the dataloader may differ)
        B = x.shape[0]

        # [B, 3, H, W] -> [B, D, H/P, W/P], [B, D, N]
        x = self.conv_layer(x).reshape((B, D, N))

        # [B, D, N] -> [B, N, D]
        x = torch.transpose(x, -2, -1)

        # prepend a classification token to every image's patch sequence
        # [B, N, D] -> [B, N+1, D]
        classification_token = self.classification_token.expand(B, -1, -1)
        x = torch.cat((x, classification_token), dim=1)

        # add positional embeddings to the patch embeddings
        x = self.dropout(x + self.positional_embeddings)

        # feed patch embeddings to the transformer encoder
        x = self.transformer_encoder(x)

        # extract encoded classification tokens
        x = torch.select(x, -2, 0).reshape(B, 1, D)

        # layer normalize encoded classification tokens
        x = self.layer_norm(x)

        # feed layer normalized classification tokens to MLP head
        x = F.tanh(self.hidden_layer(x))

        return F.softmax(self.output_layer(x), dim=-1).squeeze(1)