import torch as t
from torch import nn
import einops
from typing import Optional

from src import activations, layers

# TODO: Add MoE, allow for GQA/MQA, add RoPE and kv-caching

class MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, project_dim: int, activation: str, dropout_rate: Optional[float]=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_dim = project_dim

        if activation == 'relu':
            self.activation = activations.relu
        elif activation == 'gelu':
            self.activation = activations.gelu
        elif activation == 'swish':
            self.activation = activations.swish
        elif activation == 'sigmoid':
            self.activation = activations.sigmoid
        else:
            raise NotImplementedError("Only relu, gelu, swish, and sigmoid activation functions are implemented.")

        self.linear1 = layers.Linear(self.embed_dim, self.project_dim)
        self.linear2 = layers.Linear(self.project_dim, self.embed_dim)

        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout_layer = layers.Dropout(self.dropout_rate)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        if self.dropout_rate:
            return self.dropout_layer(x)
        else:
            return x


class GLUBlock(nn.Module):
    def __init__(self, embed_dim: int, project_dim: int, activation: str, dropout_rate: Optional[float]=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_dim = project_dim

        if activation == 'relu':
            self.activation = activations.relu
        elif activation == 'gelu':
            self.activation = activations.gelu
        elif activation == 'swish':
            self.activation = activations.swish
        elif activation == 'sigmoid':
            self.activation = activations.sigmoid
        else:
            raise NotImplementedError("Only relu, gelu, swish, and sigmoid activation functions are implemented.")

        self.linear1 = layers.Linear(self.embed_dim, self.project_dim)
        self.linear2 = layers.Linear(self.embed_dim, self.project_dim)
        self.linear3 = layers.Linear(self.project_dim, self.embed_dim)

        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout_layer = layers.Dropout(self.dropout_rate)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.activation(self.linear1(x)) * self.linear2(x)
        x = self.linear3(x)
        if self.dropout_rate:
            return self.dropout_layer(x)
        else:
            return x

class MixtureofExpertsBlock(nn.Module):
    pass

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: Optional[float]=None, rotary_embedding: bool=False, rotary_base: Optional[int]=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        # Assuming head_dim = embed_dim / num_heads
        self.head_dim = self.embed_dim//self.num_heads

        self.rotary_embedding = rotary_embedding
        self.rotary_base = rotary_base

        if self.rotary_embedding:
            if self.rotary_base is not None:
                self.rotary_layer = layers.RotaryPositionEmbedding(self.head_dim, self.rotary_base)
            else:
                self.rotary_layer = layers.RotaryPositionEmbedding(self.head_dim)

        self.linear_q = layers.Linear(embed_dim, embed_dim)
        self.linear_k = layers.Linear(embed_dim, embed_dim)
        self.linear_v = layers.Linear(embed_dim, embed_dim)
        self.linear_o = layers.Linear(embed_dim, embed_dim)

        self.dropout_rate = dropout_rate
        # Optionally has dropout layers after the attention pattern softmax and at the end of the block like in GPT-2
        if self.dropout_rate:
            self.dropout_layer_1 = layers.Dropout(self.dropout_rate)
            self.dropout_layer_2 = layers.Dropout(self.dropout_rate)
    
    def forward(self, x: t.Tensor, attention_mask: Optional[t.Tensor]=None) -> t.Tensor:
        Q = einops.rearrange(self.linear_q(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads)
        K = einops.rearrange(self.linear_k(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads)
        V = einops.rearrange(self.linear_v(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads)
    
        if self.rotary_embedding:
            Q = self.rotary_layer(Q)
            K = self.rotary_layer(K)

        pre_att_pattern = t.einsum('... h s d, ... h t d -> ... h s t', Q, K)
        pre_att_pattern /= self.head_dim ** 0.5

        if attention_mask is not None:
            pre_att_pattern += attention_mask

        att_pattern = t.softmax(pre_att_pattern, dim=-1)
        if self.dropout_rate is not None:
            att_pattern = self.dropout_layer_1(att_pattern)

        res = t.einsum('... h s t, ... h t d -> ... h s d', att_pattern, V)
        res = einops.rearrange(res, '... head s dhead -> ... s (head dhead)')
        res = self.linear_o(res)
        if self.dropout_rate is not None:
            return self.dropout_layer_2(res)
        else:
            return res

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, project_dim: int, mlp_type:str, activation: str, norm_type: str, use_pre_norm: bool=True, parallel_layers: bool=False, dropout_rate: Optional[float]=None):
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

        self.dropout_rate = dropout_rate

        self.mha_block = MultiheadAttentionBlock(embed_dim, num_heads, dropout_rate=self.dropout_rate)
        if mlp_type == 'mlpblock':
            self.mlp_block = MLPBlock(embed_dim, project_dim, activation, dropout_rate=self.dropout_rate)
        elif mlp_type == 'glublock':
            self.mlp_block = GLUBlock(embed_dim, project_dim, activation, dropout_rate=self.dropout_rate)
        else:
            raise NotImplementedError("MLP types other than mlpblock and glublock have not been implemented.")

        self.parallel_layers = parallel_layers

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Building a decoder mask where all entries above the diagonal are negative infinity and all others are zero
        seq_len = x.shape[-2]
        att_mask = t.where(t.arange(seq_len).unsqueeze(1) < t.arange(seq_len), -t.inf, 0)

        if self.parallel_layers:
            if self.use_pre_norm:
                return x + self.mha_block(self.norm_layer1(x), attention_mask=att_mask) + self.mlp_block(self.norm_layer2(x))
            else:
                # Not sure if anyone has actually done this
                return self.norm_layer1(x + self.mha_block(x, attention_mask=att_mask) + self.mlp_block(x))
        else:
            if self.use_pre_norm:
                x = x + self.mha_block(self.norm_layer1(x), attention_mask=att_mask)
                return x + self.mlp_block(self.norm_layer2(x))
            else:
                x = self.norm_layer1(x + self.mha_block(x, attention_mask=att_mask))
                return self.norm_layer2(x + self.mlp_block(x))