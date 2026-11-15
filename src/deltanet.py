import torch
import math
from torch import nn
from torch.nn import functional as F


def delta_rule_chunkwise(q, k, v, beta, chunk=64):
    batch_size, n_heads, seq_len, head_dim = q.shape
    pad = (-seq_len) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
    n_chunks = (seq_len + pad) // chunk
    shape = (batch_size, n_heads, n_chunks, chunk, head_dim)
    q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
    beta = beta.reshape(batch_size, n_heads, n_chunks, chunk, 1)
    kb = k * beta
    vb = v * beta
    strict = (kb @ k.transpose(-1, -2)).tril(-1)
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    solved = torch.linalg.solve_triangular(eye + strict, torch.cat([kb, vb], dim=-1), upper=False)
    w, u = solved[..., :head_dim], solved[..., head_dim:]
    qk = (q @ k.transpose(-1, -2)).tril()
    state = q.new_zeros(batch_size, n_heads, head_dim, head_dim)
    outputs = []
    for c in range(n_chunks):
        state_t = state.transpose(-1, -2)
        u_eff = u[:, :, c] - w[:, :, c] @ state_t
        outputs.append(q[:, :, c] @ state_t + qk[:, :, c] @ u_eff)
        state = state + u_eff.transpose(-1, -2) @ k[:, :, c]
    out = torch.stack(outputs, dim=2).reshape(batch_size, n_heads, n_chunks * chunk, head_dim)
    return out[:, :, :seq_len]

class DeltaRuleMixer(nn.Module):
    def __init__(self, d_inner, n_heads, d_conv=4, chunk=64):
        super().__init__()
        if d_inner <= 0 or n_heads <= 0 or d_inner % n_heads != 0:
            raise ValueError("d_inner must be positive and divisible by n_heads")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive")
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads
        self.qkv_proj = nn.Linear(d_inner, 3 * d_inner, bias=False)
        self.conv = nn.Conv1d(
            3 * d_inner,
            3 * d_inner,
            kernel_size=d_conv,
            groups=3 * d_inner,
            padding=d_conv - 1,
        )
        self.beta_proj = nn.Linear(d_inner, n_heads)
        self.chunk = chunk
        self.out_norm = nn.LayerNorm(self.head_dim)

    """def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = self.conv(qkv.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        state = x.new_zeros(batch_size, self.n_heads, self.head_dim, self.head_dim)
        outputs = []
        for index in range(seq_len):
            key = k[:, :, index]
            value = v[:, :, index]
            prediction = torch.einsum("bhvk,bhk->bhv", state, key)
            error = value - prediction
            rate = beta[:, :, index].unsqueeze(-1)
            state = state + rate.unsqueeze(-1) * error.unsqueeze(-1) * key.unsqueeze(-2)
            outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, :, index]))
        return torch.stack(outputs, dim=2).transpose(1, 2).reshape(batch_size, seq_len, self.d_inner)
    """
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = F.silu(self.conv(qkv.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2).float()
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2).float()
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2).float()
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).float()
        out = delta_rule_chunkwise(q, k, v, beta, self.chunk)
        out = self.out_norm(out)
        return out.transpose(1, 2).reshape(batch_size, seq_len, self.d_inner).to(x.dtype)


class DeltaNetBlock(nn.Module):
    def __init__(self, d_model, n_heads, expand=2, d_conv=4, dropout=0.1):
        super().__init__()
        if d_model <= 0 or expand <= 0 or d_conv <= 0:
            raise ValueError("DeltaNet dimensions must be positive")
        self.d_inner = d_model * expand
        if self.d_inner % n_heads != 0:
            raise ValueError("d_model * expand must be divisible by n_heads")
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.mixer = DeltaRuleMixer(self.d_inner, n_heads, d_conv)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        normalized = self.norm(x)
        signal, gate = self.in_proj(normalized).chunk(2, dim=-1)
        mixed = self.mixer(signal)
        return residual + self.dropout(self.out_proj(mixed * F.silu(gate)))


class DeltaNetLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        expand=2,
        d_conv=4,
        max_seq_len=512,
        dropout=0.1,
        pad_id=None,
    ):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or n_heads <= 0 or n_layers <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, d_model, n_heads, n_layers, and max_seq_len must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DeltaNetBlock(d_model, n_heads, expand, d_conv, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        #self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if pad_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[pad_id].zero_()
        residual_scale = 1.0 / math.sqrt(2 * n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.out_proj.weight.mul_(residual_scale)

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        x = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        if targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        loss = F.cross_entropy(
            logits.reshape(batch_size * seq_len, self.vocab_size),
            targets.reshape(batch_size * seq_len),
            ignore_index=self.pad_id if self.pad_id is not None else -100,
        )
        return logits, loss
