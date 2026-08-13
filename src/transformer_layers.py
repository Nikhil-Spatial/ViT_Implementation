import torch.nn.functional as F
from configs import HEADS, D, N
import torch.nn as nn
import torch
import math

class SelfAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

        # dimensions of each token in the query, key, and value tensors
        self.D_qkv = D // HEADS

        # Q, K, and V transformations are purely linear
        self.Q_linear_transform = nn.Linear(D, self.D_qkv, bias=False)
        self.K_linear_transform = nn.Linear(D, self.D_qkv, bias=False)
        self.V_linear_transform = nn.Linear(D, self.D_qkv, bias=False)

    def forward(self, x):
        # query, key, and value tensors
        Q = self.Q_linear_transform(x)
        K = self.K_linear_transform(x)
        V = self.V_linear_transform(x)

        # compute attention scores:
        # softmax(matmul(Q, transpose(K)) / sqrt(D_qkv))
        attention_scores = F.softmax(
            (Q @ K.transpose(-2, -1)) / math.sqrt(self.D_qkv),
            dim=-1
        )

        # compute head output
        return attention_scores @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # list of self-attention heads
        self.self_attention_heads = [SelfAttentionHead() for _ in range(HEADS)]

        # apply linear transformation to vertically concatenated head outputs
        self.output_linear_transform = nn.Linear(D, D, bias=False)

    def forward(self, x):
        # compute head outputs
        head_outputs = [self_attention_head(x) for self_attention_head
                        in self.self_attention_heads]

        # vertically concatenate the head outputs
        output = torch.cat(head_outputs, dim=-1)

        # apply linear transformation
        return self.output_linear_transform(output)

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
