import torch.nn.functional as F
from configs import HEADS, D, N
import torch.nn as nn
import torch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # dimensions of each token in the query, key, and value tensors
        self.D_qkv = D / HEADS

        # Q, K, and V transformations are purely linear
        self.Q_linear_transform = nn.Linear(D, D_qkv, bias=False)
        self.K_linear_transform = nn.Linear(D, D_qkv, bias=False)
        self.V_linear_transform = nn.Linear(D, D_qkv, bias=False)

    def forward(self, x):
        # query, key, and value tensors
        Q = self.Q_linear_transform(x)
        K = self.K_linear_transform(x)
        V = self.V_linear_transform(x)

        torch.matmul(Q, K)



class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # layer normalization layers
        self.layer_norm_1 = nn.LayerNorm(D)
        self.layer_norm_2 = nn.LayerNorm(D)



        #

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
