from configs import D, P, B, N, dropout_P
import torch.nn as nn
import torch

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        # create learnable lookup table for positional indices
        self.embedding = nn.Embedding(P, D)

        self.dropout = nn.Dropout(dropout_P)

    def forward(self, x):

