import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

# Scaled Dot Product Attention
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


# Multi Head Attention
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        # h <-> attention heads; d_model <-> hidden
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)



class GELU(nn.Module):
    """BERT used Gelu instead of Relu"""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# 前馈网络FFN
class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# 层归一化 LayerNorm
class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x): # x : [32, 12, 256]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # print(self.a_2.shape, self.b_2.shape, x.shape)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))




# transformer (encoder)
class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class InputEmbedding(nn.Module):
    """
        InputEmbedding : project the input to embedding size through a fully connected layer
    """

    def __init__(self, num_features, embedding_dim, dropout=0.1):
        """
        :param feature_num: number of input features
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.input = nn.Linear(in_features=num_features, out_features=embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embedding_dim

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)
        embed = self.input(input_sequence)  # [batch_size, seq_length, embedding_dim]

        # x = embed.repeat(1, 1, 2)           # [batch_size, seq_length, embedding_dim * 2]

        return self.dropout(embed)


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=366):
#         super().__init__()

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len+1, d_model).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)         # [max_len, 1]
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # [d_model/2,]

#         pe[1:, 0::2] = torch.sin(position * div_term)   # broadcasting to [max_len, d_model/2]
#         pe[1:, 1::2] = torch.cos(position * div_term)   # broadcasting to [max_len, d_model/2]

#         self.register_buffer('pe', pe)

#     def forward(self, doy):
#         return self.pe[doy, :]


class PositionalEncoding(nn.Module):
    '''
    The positional encoding class is used in the encoder and decoder layers.
    It's role is to inject sequence order information into the data since self-attention
    mechanisms are permuatation equivariant. Naturally, this is not required in the static
    transformer since there is no concept of 'order' in a portfolio.'''

    def __init__(self, window, d_model):
        super().__init__()

        self.register_buffer('d_model', torch.tensor(d_model, dtype = torch.float64))

        pe = torch.zeros(window, d_model)
        for pos in range(window):
            for i in range(0, d_model, 2):
              pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))

            for i in range(1, d_model, 2):
              pe[pos, i] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x * torch.sqrt(self.d_model) + self.pe[:,:x.size(1)]


# 模型拼接TVTS-BERT
class TVTSBERT(nn.Module):

    def __init__(self, num_features, hidden, n_layers, attn_heads, dropout=0.1, pe_window=100):
        """
        :num_features: number of input features
        :hidden: hidden size of the SITS-BERT model
        :n_layers: numbers of Transformer blocks (layers)
        :attn_heads: number of attention heads
        :dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        # self.embedding = InputEmbedding(num_features, int(hidden/2))
        self.embedding = InputEmbedding(num_features, int(hidden))

        self.pe = PositionalEncoding(pe_window, hidden)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        mask = (mask > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(input_sequence=x)
        x = self.pe(x)

        for transformer in self.transformer_blocks:
            # print(x.shape)
            # print(mask.shape)
            x = transformer.forward(x, mask)

        return x