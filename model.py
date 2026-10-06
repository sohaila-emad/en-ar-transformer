import math
import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) with values from 0 to seq_len-1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Create the division term for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Multiplicative factor
        self.bias = nn.Parameter(torch.zeros(features))  # Additive factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, features)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h  # Dimension of each head
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout: nn.Dropout = None):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask == 0, float("-inf")
            )

        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)  # (batch_size, seq_len, d_model)
        value = self.w_v(v)  # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        query = query.view(
            query.size(0), query.size(1), self.h, self.d_k
        ).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.size(0), value.size(1), self.h, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)

        return self.w_o(x)


class BuildResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))


class EncoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [BuildResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, mask)
        )
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [BuildResidualConnection(d_model, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src_embedded = self.src_embed(src)
        src_pos_encoded = self.src_pos(src_embedded)
        return self.encoder(src_pos_encoded, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt_embedded = self.tgt_embed(tgt)
        tgt_pos_decoded = self.tgt_pos(tgt_embedded)
        return self.decoder(
            tgt_pos_decoded, encoder_output, src_mask, tgt_mask
        )

    def project(self, x):
        return self.projection_layer(x)

    @staticmethod
    def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> "Transformer":
        # Create the embedding layers
        src_embed = InputEmbeddings(d_model, src_vocab_size)
        tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

        # Create the positional encoding layers
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        # Create encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attn = MultiHeadAttention(d_model, h, dropout)
            feed_forward = FeedForward(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(
                d_model, encoder_self_attn, feed_forward, dropout
            )
            encoder_blocks.append(encoder_block)

        # Create decoder blocks
        decoder_blocks = []
        for _ in range(N):
            decoder_self_attn = MultiHeadAttention(d_model, h, dropout)
            decoder_cross_attn = MultiHeadAttention(d_model, h, dropout)
            feed_forward = FeedForward(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(
                d_model,
                decoder_self_attn,
                decoder_cross_attn,
                feed_forward,
                dropout,
            )
            decoder_blocks.append(decoder_block)

        # Create Encoder and Decoder modules
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

        # Create projection layer
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        # Return configured Custom Transformer instance
        return Transformer(
            encoder,
            decoder,
            src_embed,
            tgt_embed,
            src_pos,
            tgt_pos,
            projection_layer,
        )

 

    