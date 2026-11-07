import math

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveStateSpace(nn.Module):
    def __init__(self, d_inner, d_state, dt_rank):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.in_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))
        nn.init.uniform_(self.dt_proj.weight, -dt_rank ** -0.5, dt_rank ** -0.5)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        parameters = self.in_proj(x)
        dt_features, B, C = torch.split(parameters, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_features))
        A = -torch.exp(self.A_log.float()).to(dtype=x.dtype)
        state = x.new_zeros(batch_size, self.d_inner, self.d_state)
        outputs = []
        for index in range(seq_len):
            delta = dt[:, index]
            delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
            delta_B = delta.unsqueeze(-1) * B[:, index].unsqueeze(1)
            state = delta_A * state + delta_B * x[:, index].unsqueeze(-1)
            y = (state * C[:, index].unsqueeze(1)).sum(dim=-1)
            outputs.append(y + self.D * x[:, index])
        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError("Mamba dimensions must be positive")
        self.d_inner = d_model * expand
        dt_rank = dt_rank or math.ceil(d_model / 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.ssm = SelectiveStateSpace(self.d_inner, d_state, dt_rank)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        x = F.silu(x)
        x = self.ssm(x)
        return residual + self.out_proj(x * F.silu(gate))


class MambaLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        max_seq_len=512,
        dropout=0.1,
        pad_id=None,
        **unused,
    ):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or n_layers <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, d_model, n_layers, and max_seq_len must be positive")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
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
