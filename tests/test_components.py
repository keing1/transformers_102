import unittest
import torch as t
import torch.nn.functional as F
from torchtune import modules
import einops
import sys
import os


from src import activations, layers, blocks

class TestTransformerComponents(unittest.TestCase):
    def test_activation_functions(self):
        x = t.randn(100)
        x2 = x.reshape((10,10))
        assert t.allclose(activations.relu(x), F.relu(x))
        assert t.allclose(activations.relu(x2), F.relu(x2))

        assert t.allclose(activations.gelu(x), F.gelu(x))
        assert t.allclose(activations.gelu(x2), F.gelu(x2))

        assert t.allclose(activations.swish(x), F.silu(x))
        assert t.allclose(activations.swish(x2), F.silu(x2))

        assert t.allclose(activations.sigmoid(x), F.sigmoid(x))
        assert t.allclose(activations.sigmoid(x2), F.sigmoid(x2))

    def test_normalizers(self):
        x = t.randn(100)
        x2 = x.reshape((10,10))

        W1 = t.nn.Parameter(t.randn(100))
        b1 = t.nn.Parameter(t.randn(100))
        W2 = t.nn.Parameter(t.randn(10))
        b2 = t.nn.Parameter(t.randn(10))

        ln1 = layers.LayerNorm(x.shape[-1:])
        torch_ln1 = t.nn.LayerNorm(x.shape[-1:])
        ln2 = layers.LayerNorm(x2.shape[-1:])
        torch_ln2 = t.nn.LayerNorm(x2.shape[-1:])

        assert t.allclose(ln1.weight, torch_ln1.weight)
        assert t.allclose(ln2.weight, torch_ln2.weight)
        
        ln1.weight, ln1.bias = W1, b1
        torch_ln1.weight, torch_ln1.bias = W1, b1
        ln2.weight, ln2.bias = W2, b2
        torch_ln2.weight, torch_ln2.bias = W2, b2

        assert t.allclose(ln1(x), torch_ln1(x), atol=1e-7)
        assert t.allclose(ln2(x2), torch_ln2(x2), atol=1e-7)

        rn1 = layers.RMSNorm(x.shape[-1:])
        torch_rn1 = t.nn.RMSNorm(x.shape[-1:])
        rn2 = layers.RMSNorm(x2.shape[-1:])
        torch_rn2 = t.nn.RMSNorm(x2.shape[-1:])

        assert t.allclose(rn1.weight, torch_rn1.weight)
        assert t.allclose(rn2.weight, torch_rn2.weight)
        
        rn1.weight, rn1.bias = W1, b1
        torch_rn1.weight, torch_rn1.bias = W1, b1
        rn2.weight, rn2.bias = W2, b2
        torch_rn2.weight, torch_rn2.bias = W2, b2

        assert t.allclose(rn1(x), torch_rn1(x))
        assert t.allclose(rn2(x2), torch_rn2(x2))


    def test_embedding(self):
        num_embeddings = 10
        embedding_dim = 5

        W = t.nn.Parameter(t.randn((num_embeddings, embedding_dim)))

        emb = layers.Embedding(num_embeddings, embedding_dim)
        torch_emb = t.nn.Embedding(num_embeddings, embedding_dim)

        emb.weight = W
        torch_emb.weight = W

        tokens = t.tensor([8,6,2,9])
        tokens2 = t.tensor([[5,9], [7,2]])

        assert t.allclose(emb(tokens), torch_emb(tokens))
        assert t.allclose(emb(tokens2), torch_emb(tokens2))


    def test_linear(self):
        in_dim = 5
        out_dim = 10
        W = t.nn.Parameter(t.randn((out_dim, in_dim)))
        b = t.nn.Parameter(t.randn((out_dim,)))

        lin = layers.Linear(in_dim, out_dim)
        torch_lin = t.nn.Linear(in_dim, out_dim)

        lin.weight, lin.bias = W, b
        torch_lin.weight, torch_lin.bias = W, b

        x = t.arange(5).float()
        x2 = t.arange(20).reshape((4,5)).float()

        assert t.allclose(lin(x), torch_lin(x), atol=1e-7)
        assert t.allclose(lin(x2), torch_lin(x2), atol=1e-7)
    
    def test_rope(self):
        embed = 16
        base = 1000

        my_rpe = layers.RotaryPositionEmbedding(embed, base=base)
        torch_rpe = modules.RotaryPositionalEmbeddings(embed, base=base)

        # the torchtune rpe class assumes the sequence dimension is 2nd while my class assumes it is 3rd (specifically 2nd to last)
        x = t.arange(768).float().reshape((2,8,3,16))
        t_x = einops.rearrange(x, 'b h s d -> b s h d')

        assert t.allclose(my_rpe(x), einops.rearrange(torch_rpe(t_x), 'b s h d -> b h s d'))

    def test_attention_block(self):
        embed_dim = 10
        num_heads = 2

        Wq = t.nn.Parameter(t.randn((embed_dim, embed_dim)))
        Wk = t.nn.Parameter(t.randn((embed_dim, embed_dim)))
        Wv = t.nn.Parameter(t.randn((embed_dim, embed_dim)))
        Wo = t.nn.Parameter(t.randn((embed_dim, embed_dim)))

        bq = t.nn.Parameter(t.randn((embed_dim,)))
        bk = t.nn.Parameter(t.randn((embed_dim,)))
        bv = t.nn.Parameter(t.randn((embed_dim,)))
        bo = t.nn.Parameter(t.randn((embed_dim,)))

        t_linq = layers.Linear(embed_dim, embed_dim)
        t_link = layers.Linear(embed_dim, embed_dim)
        t_linv = layers.Linear(embed_dim, embed_dim)

        t_linq.weight, t_linq.bias = Wq, bq
        t_link.weight, t_link.bias = Wk, bk
        t_linv.weight, t_linv.bias = Wv, bv

        x = t.arange(40).reshape((4,10)).float()
        x2 = t.arange(80).reshape((2,4,10)).float()

        attn_mask = t.where(t.arange(4).unsqueeze(1) < t.arange(4), -t.inf, 0)

        mha = blocks.MultiheadAttentionBlock(embed_dim, num_heads, attn_bias=True)
        mha.linear_q.weight, mha.linear_q.bias = Wq, bq
        mha.linear_k.weight, mha.linear_k.bias = Wk, bk
        mha.linear_v.weight, mha.linear_v.bias = Wv, bv
        mha.linear_o.weight, mha.linear_o.bias = Wo, bo

        t_mha = t.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        t_mha.in_proj_weight, t_mha.in_proj_bias = t.nn.Parameter(t.cat((Wq, Wk, Wv), dim=0)), t.nn.Parameter(t.cat((bq, bk, bv), dim=0))
        t_mha.out_proj.weight, t_mha.out_proj.bias = Wo, bo

        assert t.allclose(mha(x, attention_mask=attn_mask), t_mha(x, x, x, attn_mask=attn_mask)[0])
        assert t.allclose(mha(x2, attention_mask=attn_mask), t_mha(x2, x2, x2, attn_mask=attn_mask)[0])

    def test_mlp_blocks(self):
        embed_dim = 10
        project_dim = 40
        activation = 'relu'

        W1 = t.nn.Parameter(t.randn((project_dim, embed_dim)))
        b1 = t.nn.Parameter(t.randn(project_dim,))
        W2 = t.nn.Parameter(t.randn((embed_dim, project_dim)))
        b2 = t.nn.Parameter(t.randn((embed_dim,)))

        mlp = blocks.MLPBlock(embed_dim, project_dim, activation)
        torch_lin1 = t.nn.Linear(embed_dim, project_dim)
        torch_lin2 = t.nn.Linear(project_dim, embed_dim)
        
        mlp.linear1.weight, mlp.linear1.bias = W1, b1
        torch_lin1.weight, torch_lin1.bias = W1, b1
        mlp.linear2.weight, mlp.linear2.bias = W2, b2
        torch_lin2.weight, torch_lin2.bias = W2, b2

        torch_mlp = t.nn.Sequential(torch_lin1, t.nn.ReLU(), torch_lin2)
        
        x = t.arange(10).float()
        x2 = t.arange(20).reshape((2,10)).float()

        assert t.allclose(mlp(x), torch_mlp(x), atol=1e-7)
        assert t.allclose(mlp(x2), torch_mlp(x2), atol=1e-7)
        
        # W1 = t.nn.Parameter(t.randn((project_dim, embed_dim)))
        # b1 = t.nn.Parameter(t.randn(project_dim,))
        # W2 = t.nn.Parameter(t.randn((project_dim, embed_dim)))
        # b2 = t.nn.Parameter(t.randn(project_dim,))
        # W3 = t.nn.Parameter(t.randn((embed_dim, project_dim)))
        # b3 = t.nn.Parameter(t.randn((embed_dim,)))

        # glu = blocks.GLUBlock(embed_dim, project_dim, activation)
        # torch_lin1 = t.nn.Linear(embed_dim, project_dim)
        # torch_lin2 = t.nn.Linear(embed_dim, project_dim)
        # torch_lin3 = t.nn.Linear(project_dim, embed_dim)
        
        # TODO: Add GLU and MoE test
        pass

    def test_transformer_block(self):
        pass

if __name__ == '__main__':
    unittest.main()