import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.qkv_bias = nn.Parameter(torch.zeros(3 * d_model))
        self.out_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.out_bias = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.xavier_uniform_(self.out_weight)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return attended @ self.out_weight.T + self.out_bias


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        x = torch.nn.functional.gelu(x @ self.w1.T + self.b1)
        x = self.dropout(x)
        return x @ self.w2.T + self.b2


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.residual_dropout(self.attention(self.norm1(x)))
        x = x + self.residual_dropout(self.feed_forward(self.norm2(x)))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=None,
        max_seq_len=512,
        dropout=0.1,
        pad_id=None,
    ):
        super().__init__()
        if vocab_size <= 0 or max_seq_len <= 0 or n_layers <= 0:
            raise ValueError("vocab_size, max_seq_len, and n_layers must be positive")
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        hidden_dim = d_ff if d_ff is not None else 4 * d_model
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        if targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(batch_size * seq_len, self.vocab_size),
            targets.reshape(batch_size * seq_len),
            ignore_index=self.pad_id if self.pad_id is not None else -100,
        )
        return logits, loss
