import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .deltanet import DeltaNetBlock
    from .mamba import MambaBlock
except ImportError:
    from deltanet import DeltaNetBlock
    from mamba import MambaBlock


class HybridLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_state=16,
        d_conv=4,
        expand=2,
        max_seq_len=512,
        dropout=0.1,
        pad_id=None,
        layer_pattern=None,
        **unused,
    ):
        super().__init__()
        pattern = layer_pattern or ("MD" * n_layers)[:n_layers]
        if not pattern or set(pattern) - {"M", "D"}:
            raise ValueError("layer_pattern must be a non-empty string of M and D")
        if layer_pattern and len(layer_pattern) != n_layers:
            raise ValueError("layer_pattern length must equal n_layers")
        if vocab_size <= 0 or d_model <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, d_model, and max_seq_len must be positive")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.layer_pattern = pattern
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(dropout)
        blocks = []
        for kind in pattern:
            if kind == "M":
                blocks.append(MambaBlock(d_model, d_state, d_conv, expand))
            else:
                blocks.append(DeltaNetBlock(d_model, n_heads, expand, d_conv, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if pad_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[pad_id].zero_()
        residual_scale = 1.0 / math.sqrt(2 * len(pattern))
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
            if self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
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