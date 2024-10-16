import torch as t

from src import layers, blocks

class GPT2Embedding(t.nn.Module):
    def __init__(self, num_token_embeddings: int, embedding_dim: int, context_length: int):
        super().__init__()
        self.num_token_embeddings = num_token_embeddings
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.position_embedding_layer = layers.Embedding(self.context_length, self.embedding_dim)
        self.token_embedding_layer = layers.Embedding(self.num_token_embeddings, self.embedding_dim)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        pos_tensor = t.arange(x.shape[-1])
        x_pos = self.position_embedding_layer(pos_tensor)
        x_tok = self.token_embedding_layer(x)
        return x_tok + x_pos

class GPT2Model(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens = 50257
        self.hidden_size = 768
        self.num_layers = 12
        self.activation_function = 'gelu'
        self.context_length = 1024
        self.dropout_rate = 0.1
        self.num_heads = 12

        self.init_dropout_layer = layers.Dropout(self.dropout_rate)
        self.embedding_layer = GPT2Embedding(self.num_tokens, self.hidden_size, self.context_length)
        self.transformer_blocks = [blocks.TransformerDecoderBlock(self.hidden_size, self.num_heads, 4*self.hidden_size, 'mlpblock', self.activation_function, 'layer_norm', use_pre_norm=True, dropout_rate=self.dropout_rate) for _ in range(self.num_layers)]
        self.final_ln = layers.LayerNorm((self.hidden_size,))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.embedding_layer(x)
        x = self.init_dropout_layer(x)
        # TODO: Add dropout option in transformer subblocks and block
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_ln(x)
        # Tied unembedding
        x = t.einsum('th, ...h -> ...t', self.embedding_layer.token_embedding_layer, x)

        return x