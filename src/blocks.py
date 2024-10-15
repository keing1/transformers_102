import torch as t
from torch import nn
import einops

from src import activations, layers

# TODO: Add MoE, allow for GQA/MQA, add RoPE and kv-caching

class MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, project_dim: int, activation: str):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_dim = project_dim

        if activation == 'relu':
            self.activation = activations.relu
        elif activation == 'gelu':
            self.activation = activations.gelu
        elif activation == 'swish':
            self.activation = activations.swish
        else:
            raise NotImplementedError("Only relu, gelu, and swish activation functions are implemented.")

        self.linear1 = layers.Linear(self.embed_dim, self.project_dim)
        self.linear2 = layers.Linear(self.project_dim, self.embed_dim)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class GLUBlock(nn.Module):
    def __init__(self, embed_dim: int, project_dim: int, activation: str):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_dim = project_dim

        if activation == 'relu':
            self.activation = activations.relu
        elif activation == 'gelu':
            self.activation = activations.gelu
        elif activation == 'swish':
            self.activation = activations.swish
        else:
            raise NotImplementedError("Only relu, gelu, and swish activation functions are implemented.")

        self.linear1 = layers.Linear(self.embed_dim, self.project_dim)
        self.linear2 = layers.Linear(self.embed_dim, self.project_dim)
        self.linear3 = layers.Linear(self.project_dim, self.embed_dim)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.activation(self.linear1(x)) * self.linear2(x)
        return self.linear3(x)

class MoEBlock(nn.Module):
    pass

class MultiheadAttentionBlock(nn.Module):
    # Includes the calculation of Q, K, and V
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        # Assuming head_dim = embed_dim / num_heads
        self.head_dim = self.embed_dim//self.num_heads

        self.linear_q = layers.Linear(embed_dim, embed_dim)
        self.linear_k = layers.Linear(embed_dim, embed_dim)
        self.linear_v = layers.Linear(embed_dim, embed_dim)
        self.linear_o = layers.Linear(embed_dim, embed_dim)
    
    def forward(self, x: t.Tensor, decoder: bool=True) -> t.Tensor:
        Q = einops.rearrange(self.linear_q(x), 'b s (head d_head) -> b head s d_head', n=self.num_heads)
        K = einops.rearrange(self.linear_k(x), 'b s (head d_head) -> b head s d_head', n=self.num_heads)
        V = einops.rearrange(self.linear_v(x), 'b s (head d_head) -> b head s d_head', n=self.num_heads)

        pre_att_pattern = t.einsum('b head s_q d_head, b head s_k d_head -> b head s_q s_k', Q, K)
        pre_att_pattern /= self.head_dim ** 0.5
        if decoder:
            # Building a decoder mask where all entries above the diagonal are negative infinity and all others are zero
            seq_len = pre_att_pattern.shape[-1]
            att_mat = t.where(t.arange(seq_len).unsqueeze(1) < t.arange(seq_len), -t.inf, 0)
            pre_att_pattern += att_mat

        att_pattern = t.softmax(pre_att_pattern, dim=-1)

        res = t.einsum('b head s_q s, b head s d_head -> b head s_q d_head', att_pattern, V)
        res = einops.rearrange(res, 'b head s d_head -> b s (head d_head)')
        return self.linear_o(res)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, project_dim: int, mlp_type:str, activation: str, norm_type: str, use_pre_norm: bool=True, parallel_layers: bool=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.project_dim = project_dim
        self.activation = activation

        if norm_type == 'layer_norm':
            self.norm_type = layers.LayerNorm
        elif norm_type == 'rms_norm':
            self.norm_type = layers.RMSNorm
        else:
            raise NotImplementedError("Normalization methods other than layer_norm and rms_norm have not been implemented.")
        self.norm_layer1 = self.norm_type((self.embed_dim,))
        if not parallel_layers or use_pre_norm:
            self.norm_layer2 = self.norm_type((self.embed_dim,))
        self.use_pre_norm = use_pre_norm

        self.mha_block = MultiheadAttentionBlock(embed_dim, num_heads)
        if mlp_type == 'mlpblock':
            self.mlp_block = MLPBlock(embed_dim, project_dim, activation)
        elif mlp_type == 'glublock':
            self.mlp_block = GLUBlock(embed_dim, project_dim, activation)
        else:
            raise NotImplementedError("MLP types other than mlpblock and glublock have not been implemented.")

        self.parallel_layers = parallel_layers

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.parallel_layers:
            if self.use_pre_norm:
                return x + self.mha_block(self.norm_layer1(x)) + self.mlp_block(self.norm_layer2(x))
            else:
                # Not sure if anyone has actually done this
                return self.norm_layer1(x + self.mha_block(x) + self.mlp_block(x))
        else:
            if self.use_pre_norm:
                return x + self.mlp_block(self.norm_layer2(x + self.mha_block(self.norm_layer1(x))))
            else:
                return self.norm_layer2(x + self.mlp_block(self.norm_layer1(x + self.mha_block(x))))