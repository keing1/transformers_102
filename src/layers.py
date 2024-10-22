import torch as t
from torch import nn
from typing import Tuple
import einops

class LayerNorm(nn.Module):
    def __init__(self, data_shape: Tuple[int, ...], eps: float=1e-5, includes_bias: bool=True):
        super().__init__()
        self.data_shape = data_shape
        self.eps = eps
        self.includes_bias = includes_bias

        self.weight = nn.Parameter(t.ones(self.data_shape))
        if self.includes_bias:
            self.bias = nn.Parameter(t.zeros(self.data_shape))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        # Calcualte mean and variance over all dimensions but the first
        agg_dims = tuple(range(len(x.shape)-len(self.data_shape), len(x.shape)))
        x_mean = t.mean(x, dim=agg_dims, keepdim=True)
        x_var = t.var(x, dim=agg_dims, keepdim=True, correction=0)

        normed_x = (x - x_mean)/(t.sqrt(x_var+self.eps))

        if self.includes_bias:
            out_x = normed_x * self.weight + self.bias
        else:
            out_x = normed_x * self.weight
        return out_x

class RMSNorm(nn.Module):
    def __init__(self, data_shape: Tuple[int, ...], eps: float=0):
        super().__init__()
        self.data_shape = data_shape
        self.eps = eps

        self.weight = nn.Parameter(t.ones(self.data_shape))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        # Calculate mean and variance over all dimensions but the first
        agg_dims = tuple(range(len(x.shape)-len(self.data_shape), len(x.shape)))

        x_ms = t.mean(x ** 2, dim=agg_dims, keepdim=True)
        normed_x = x/t.sqrt(x_ms+self.eps)
        return normed_x * self.weight
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Using default pytorch initialization
        self.weight = nn.Parameter(t.randn(num_embeddings, embedding_dim))

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.weight[x]

class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, includes_bias: bool=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.includes_bias = includes_bias

        # Using default pytorch initialization
        bound_val = 1/self.in_dim**0.5
        self.weight = nn.Parameter(t.rand(self.out_dim, self.in_dim) * 2*bound_val - bound_val)
        if self.includes_bias:
            self.bias = nn.Parameter(t.rand(self.out_dim) * 2*bound_val - bound_val)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = t.einsum('oi,...i-> ...o', self.weight, x)
        if self.includes_bias:
            x += self.bias
        return x

class Dropout(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training:
            random_mask = t.rand(x.shape) > self.dropout_rate
            x *= random_mask
            x *= 1/(1-self.dropout_rate)
            return x
        else:
            return x

# Assumes
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, embed_dim: int, base: int=10000):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.base = base

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Assumes x has dimensions (..., s, d) where s is sequence and d is the embedding dimensions, other dimensions are treated
        # as batch dimensions
        # TODO: Include positional input for RoPE at inference
        assert x.shape[-1] == self.embed_dim

        # Applying sparse product from section 3.4.2: https://arxiv.org/pdf/2104.09864
        x_split = einops.rearrange(x, '... (g p) -> ... g p', p=2)
        flipped_x = einops.rearrange(t.flip(x_split.float() * t.tensor([1, -1]), dims=(-1,)), '... g p -> ... (g p)')
        
        theta_vec = t.pow(10000, (-2 * (t.arange(self.embed_dim) // 2)) / self.embed_dim)
        seq_len = x.shape[-2]
        theta_prod = theta_vec * t.arange(seq_len).unsqueeze(dim=1)

        rot_vec_cos = t.cos(theta_prod)
        rot_vec_sin = t.sin(theta_prod)

        return rot_vec_cos * x + rot_vec_sin * flipped_x