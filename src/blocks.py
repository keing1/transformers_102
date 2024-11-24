import torch as t
from torch import nn
import einops
from typing import Optional, Tuple
from src import activations, layers


def retrieve_activation_function(activation: str):
    name_function_mapping = {
        'relu': activations.relu,
        'gelu': activations.gelu,
        'swish': activations.swish,
        'sigmoid': activations.sigmoid
    }
    try:
        act_fun = name_function_mapping[activation]
        return act_fun
    except KeyError:
        raise NotImplementedError("Only relu, gelu, swish, and sigmoid activation functions are implemented.")


class MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, project_dim: int, activation: str, dropout_rate: Optional[float]=None, includes_bias: bool=True):
        super().__init__()
        self.activation = retrieve_activation_function(activation)

        self.linear1 = layers.Linear(embed_dim, project_dim, includes_bias=includes_bias)
        self.linear2 = layers.Linear(project_dim, embed_dim, includes_bias=includes_bias)

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
    def __init__(self, embed_dim: int, project_dim: int, activation: str, dropout_rate: Optional[float]=None, includes_bias: bool=True):
        super().__init__()
        self.activation = retrieve_activation_function(activation)

        self.linear_gate = layers.Linear(embed_dim, project_dim, includes_bias=includes_bias)
        self.linear_up = layers.Linear(embed_dim, project_dim, includes_bias=includes_bias)
        self.linear_down = layers.Linear(project_dim, embed_dim, includes_bias=includes_bias)

        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout_layer = layers.Dropout(self.dropout_rate)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.activation(self.linear_gate(x)) * self.linear_up(x)
        x = self.linear_down(x)
        if self.dropout_rate:
            return self.dropout_layer(x)
        else:
            return x


class MixtureofExpertsBlock(nn.Module):
    def __init__(self, num_experts: int, num_experts_used: int, embed_dim: int, project_dim: int, activation: str, includes_bias: bool=True):
        super().__init__()
        self.activation = retrieve_activation_function(activation)
        
        self.expert_weights_up = nn.Parameter(t.randn((num_experts, project_dim, embed_dim)))
        self.expert_weights_down = nn.Parameter(t.randn((num_experts, embed_dim, project_dim)))

        self.includes_bias = includes_bias

        if self.includes_bias:
            self.expert_biases_up = nn.Parameter(t.randn((num_experts, project_dim)))
            self.expert_biases_down = nn.Parameter(t.randn((num_experts, embed_dim)))
        
        self.router = layers.Linear(embed_dim, num_experts, includes_bias=False)
        self.num_experts_used = num_experts_used
    
    def _select_top_k_expert_params(self, indices: t.Tensor, s: int) -> Tuple[t.Tensor, Optional[t.Tensor], t.Tensor, Optional[t.Tensor]]:
        k = self.num_experts_used

        weights_up_sparse = einops.rearrange(self.expert_weights_up[indices.reshape(-1)], '(b s k) u d -> b s k u d', s=s, k=k)
        if self.includes_bias:
            biases_up_sparse = einops.rearrange(self.expert_biases_up[indices.reshape(-1)], '(b s k) u -> b s k u', s=s, k=k)
        else:
            biases_up_sparse = None

        weights_down_sparse = einops.rearrange(self.expert_weights_down[indices.reshape(-1)], '(b s k) d u -> b s k d u', s=s, k=k)
        if self.includes_bias:
            biases_down_sparse = einops.rearrange(self.expert_biases_down[indices.reshape(-1)], '(b s k) d -> b s k d', s=s, k=k)
        else:
            biases_down_sparse = None
        
        return weights_up_sparse, biases_up_sparse, weights_down_sparse, biases_down_sparse

    def forward(self, x: t.Tensor) -> t.Tensor:
        s = x.shape[-2]
        k = self.num_experts_used

        router_logits = self.router(x)
        router_logits, indices = t.topk(router_logits, k)
        router_weights = t.softmax(router_logits, dim=-1)

        # Select weights and biases from active experts
        weights_up, biases_up, weights_down, biases_down = self._select_top_k_expert_params(indices, s)

        x = t.einsum('bskud,bsd->bsku', weights_up,x)
        if self.includes_bias:
            x = x + biases_up
        x = self.activation(x)
        x = t.einsum('bskdu,bsku->bskd', weights_down, x)
        if self.includes_bias:
            x = x + biases_down
        x = t.einsum('bskd,bsk->bsd', x, router_weights)

        return x


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, rotary_embedding: bool=False, rotary_base: Optional[int]=None, rope_alternate: bool=False, includes_bias: bool=False, dropout_rate: Optional[float]=None):
        super().__init__()
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        # Assuming head_dim = embed_dim / num_heads
        self.head_dim = embed_dim//self.num_heads

        self.rotary_embedding = rotary_embedding
        self.rope_alternate = rope_alternate

        if self.rotary_embedding:
            if rotary_base is not None:
                self.rotary_layer = layers.RotaryPositionEmbedding(self.head_dim, rotary_base)
            else:
                self.rotary_layer = layers.RotaryPositionEmbedding(self.head_dim)

        self.linear_q = layers.Linear(embed_dim, embed_dim, includes_bias=includes_bias)
        self.linear_k = layers.Linear(embed_dim, embed_dim, includes_bias=includes_bias)
        self.linear_v = layers.Linear(embed_dim, embed_dim, includes_bias=includes_bias)
        self.linear_o = layers.Linear(embed_dim, embed_dim, includes_bias=includes_bias)

        self.dropout_rate = dropout_rate
        # Optionally has dropout layers after the attention pattern softmax and at the end of the block like in GPT-2
        if self.dropout_rate:
            self.dropout_layer_1 = layers.Dropout(self.dropout_rate)
            self.dropout_layer_2 = layers.Dropout(self.dropout_rate)
    
    def forward(self, x: t.Tensor, attention_mask: Optional[t.Tensor]=None, kv_cache: Optional[Tuple[t.Tensor, t.Tensor]]=None) -> t.Tensor:
        Q = einops.rearrange(self.linear_q(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads)
        K = einops.rearrange(self.linear_k(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads)
        V = einops.rearrange(self.linear_v(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads)
    
        if self.rotary_embedding:
            start_index = 0 if kv_cache is None else kv_cache[0].shape[-2]
            Q = self.rotary_layer(Q, rope_alternate=self.rope_alternate, start_index=start_index)
            K = self.rotary_layer(K, rope_alternate=self.rope_alternate, start_index=start_index)
        
        if kv_cache is not None:
            (K_cache, V_cache) = kv_cache
            K = t.cat([K_cache, K], dim=-2)
            V = t.cat([V_cache, V], dim=-2)

        pre_att_pattern = t.einsum('... h s d, ... h t d -> ... h s t', Q, K)
        pre_att_pattern = pre_att_pattern / self.head_dim ** 0.5

        if attention_mask is not None:
            pre_att_pattern = pre_att_pattern + attention_mask

        att_pattern = t.softmax(pre_att_pattern, dim=-1)
        if self.dropout_rate is not None:
            att_pattern = self.dropout_layer_1(att_pattern)

        out = t.einsum('... h s t, ... h t d -> ... h s d', att_pattern, V)
        out = einops.rearrange(out, '... head s dhead -> ... s (head dhead)')
        out = self.linear_o(out)
        if self.dropout_rate is not None:
            out = self.dropout_layer_2(out)
        
        return out, (K, V)


class GQABlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads_q: int, num_heads_kv: int, rotary_embedding: bool=False, rotary_base: Optional[int]=None):
        super().__init__()
        self.num_heads_q = num_heads_q

        assert embed_dim % num_heads_q == 0
        # Assuming head_dim = embed_dim / num_heads
        self.head_dim = embed_dim // self.num_heads_q

        self.num_heads_kv = num_heads_kv
        assert num_heads_q % num_heads_kv == 0
        self.head_ratio = num_heads_q // num_heads_kv

        self.rotary_embedding = rotary_embedding

        if self.rotary_embedding:
            if rotary_base is not None:
                self.rotary_layer = layers.RotaryPositionEmbedding(self.head_dim, rotary_base)
            else:
                self.rotary_layer = layers.RotaryPositionEmbedding(self.head_dim)

        self.linear_q = layers.Linear(embed_dim, embed_dim, includes_bias=False)
        self.linear_k = layers.Linear(embed_dim, num_heads_kv*self.head_dim, includes_bias=False)
        self.linear_v = layers.Linear(embed_dim, num_heads_kv*self.head_dim, includes_bias=False)
        self.linear_o = layers.Linear(embed_dim, embed_dim, includes_bias=False)

    def forward(self, x: t.Tensor, attention_mask: Optional[t.Tensor]=None, kv_cache: Optional[Tuple[t.Tensor, t.Tensor]]=None) -> t.Tensor:
        Q = einops.rearrange(self.linear_q(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads_q)
        K = einops.rearrange(self.linear_k(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads_kv)
        V = einops.rearrange(self.linear_v(x), '... s (head dhead) -> ... head s dhead', head=self.num_heads_kv)
    
        if self.rotary_embedding:
            start_index = 0 if kv_cache is None else kv_cache[0].shape[-2]
            Q = self.rotary_layer(Q, start_index=start_index)
            K = self.rotary_layer(K, start_index=start_index)

        if kv_cache is not None:
            (K_cache, V_cache) = kv_cache
            K = t.cat([K_cache, K], dim=-2)
            V = t.cat([V_cache, V], dim=-2)

        Q = einops.rearrange(Q, '... (head group) s dhead -> ... head group s dhead', group=self.num_heads_kv)

        pre_att_pattern = t.einsum('... h g s d, ... g t d -> ... h g s t', Q, K)
        pre_att_pattern = pre_att_pattern / self.head_dim ** 0.5

        if attention_mask is not None:
            pre_att_pattern = pre_att_pattern + attention_mask

        att_pattern = t.softmax(pre_att_pattern, dim=-1)

        res = t.einsum('... h g s t, ... g t d -> ... h g s d', att_pattern, V)
        res = einops.rearrange(res, '... head group s dhead -> ... s (head group dhead)')
        return self.linear_o(res), (K, V)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, project_dim: int, activation: str, norm_type: str, num_heads_kv: Optional[int]=None, mlp_type: str='mlpblock', mha_type: str='mhablock', use_pre_norm: bool=True, parallel_layers: bool=False, dropout_rate: Optional[float]=None, rotary_embedding: bool=False, rotary_base: Optional[int]=None, rope_alternate: bool=False, mha_bias: bool=False, mlp_bias: bool=True, num_experts: Optional[int]=None, num_experts_used:Optional[int]=None):
        super().__init__()
        self.num_heads = num_heads
        self.activation = activation

        if norm_type == 'layer_norm':
            self.norm_type = layers.LayerNorm
        elif norm_type == 'rms_norm':
            self.norm_type = layers.RMSNorm
        else:
            raise NotImplementedError("Normalization methods other than layer_norm and rms_norm have not been implemented.")
        self.norm_layer1 = self.norm_type((embed_dim,))
        if not parallel_layers or use_pre_norm:
            self.norm_layer2 = self.norm_type((embed_dim,))
        self.use_pre_norm = use_pre_norm

        self.dropout_rate = dropout_rate

        self.rotary_embedding = rotary_embedding


        if mha_type == 'mhablock':
            self.mha_block = MultiheadAttentionBlock(embed_dim, num_heads, dropout_rate=self.dropout_rate, rotary_embedding=self.rotary_embedding, rotary_base=rotary_base, rope_alternate=rope_alternate, includes_bias=mha_bias)
        elif mha_type == 'gqablock':
            self.mha_block = GQABlock(embed_dim, num_heads, num_heads_kv, rotary_embedding=self.rotary_embedding, rotary_base=rotary_base)
        else:
            raise NotImplementedError("Attention types other than mhablock and gqablock have not been implemented.")

        if mlp_type == 'mlpblock':
            self.mlp_block = MLPBlock(embed_dim, project_dim, activation, dropout_rate=self.dropout_rate, includes_bias=mlp_bias)
        elif mlp_type == 'glublock':
            self.mlp_block = GLUBlock(embed_dim, project_dim, activation, dropout_rate=self.dropout_rate, includes_bias=mlp_bias)
        elif mlp_type == 'moeblock':
            assert num_experts is not None
            assert num_experts_used is not None
            self.mlp_block = MixtureofExpertsBlock(num_experts, num_experts_used, embed_dim, project_dim, activation, includes_bias=mlp_bias)
        else:
            raise NotImplementedError("MLP types other than mlpblock, glublock, and moeblock have not been implemented.")

        self.parallel_layers = parallel_layers

    def forward(self, x: t.Tensor, kv_cache: Optional[Tuple[t.Tensor, t.Tensor]]=None) -> t.Tensor:
        # Building a decoder mask where all entries above the diagonal are negative infinity and all others are zero
        total_seq_len = x.shape[-2]
        if kv_cache is not None:
            cache_len = kv_cache[0].shape[-2]
            total_seq_len += cache_len
        att_mask = t.where(t.arange(total_seq_len).unsqueeze(1) < t.arange(total_seq_len), -t.inf, 0)
        if kv_cache is not None:
            att_mask = att_mask[cache_len:]

        if self.parallel_layers:
            if self.use_pre_norm:
                att_out, kv_cache = self.mha_block(self.norm_layer1(x), attention_mask=att_mask, kv_cache=kv_cache)
                return x + att_out + self.mlp_block(self.norm_layer2(x)), kv_cache
            else:
                # Not sure if anyone has actually done this
                att_out, kv_cache = self.mha_block(x, attention_mask=att_mask, kv_cache=kv_cache)
                return self.norm_layer1(x + att_out + self.mlp_block(x)), kv_cache
        else:
            if self.use_pre_norm:
                att_out, kv_cache = self.mha_block(self.norm_layer1(x), attention_mask=att_mask, kv_cache=kv_cache)
                x = x + att_out
                return x + self.mlp_block(self.norm_layer2(x)), kv_cache
            else:
                att_out, kv_cache = self.mha_block(x, attention_mask=att_mask, kv_cache=kv_cache)
                x = self.norm_layer1(x + att_out)
                return self.norm_layer2(x + self.mlp_block(x)), kv_cache