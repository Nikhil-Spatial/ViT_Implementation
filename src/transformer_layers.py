import torch.nn.functional as F
from configs import HEADS, D, N
import torch.nn as nn
import torch
import math

class SelfAttentionHead(nn.Module):
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

        # compute attention scores:
        # softmax(matmul(Q, transpose(K)) / sqrt(D_qkv))
        attention_scores = F.softmax(
            (Q @ K.transpose(-2, -1)) / math.sqrt(self.D_qkv)
        )

        # compute head output
        return attention_scores @ V

    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_attention_head_1 = SelfAttentionHead()
        self.self_attention_head_2 = SelfAttentionHead()
        self.self_attention_head_3 = SelfAttentionHead()
        self.self_attention_head_4 = SelfAttentionHead()


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
