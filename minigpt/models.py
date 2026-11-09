"""Attention, FFN và decoder block tự viết; chỉ dùng các phép toán/layer cơ bản."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import BOS, PAD


def output_logits(logits):
    # Same prediction support for all three LMs: never predict PAD or BOS.
    blocked = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
    blocked[PAD] = blocked[BOS] = True
    return logits.masked_fill(blocked, float("-inf"))


def token_loss(logits, targets, reduction="mean"):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=PAD, reduction=reduction)


class CausalAttention(nn.Module):
    def __init__(self, dim, heads, context, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("dim phải chia hết cho heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal", torch.tril(torch.ones(context, context, dtype=torch.bool)), persistent=False)

    def forward(self, x, valid):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self.causal[:t, :t][None, None] & valid[:, None, None, :]
        # Right padding only; padded queries need finite softmax even for all-PAD rows.
        eye = torch.eye(t, dtype=torch.bool, device=x.device)[None, None]
        allowed = allowed | ((~valid[:, None, :, None]) & eye)
        weights = F.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        out = self.dropout(weights) @ v
        out = out.transpose(1, 2).contiguous().reshape(b, t, d)
        return self.dropout(self.proj(out)) * valid.unsqueeze(-1)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, context, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = CausalAttention(dim, heads, context, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, valid):
        x = x + self.attention(self.ln1(x), valid)
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, context=256, dim=384, layers=6, heads=6, dropout=.1, tie_weights=True):
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.position = nn.Embedding(context, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(dim, heads, context, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init)
        if tie_weights:
            self.head.weight = self.embedding.weight

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.size(1) > self.context:
            raise ValueError("Chuỗi dài hơn positional embedding")
        valid = ids.ne(PAD)
        x = self.dropout(self.embedding(ids) + self.position(torch.arange(ids.size(1), device=ids.device)))
        for block in self.blocks:
            x = block(x, valid)
        return output_logits(self.head(self.norm(x)))


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, context=256, embedding_dim=256, hidden_dim=512, layers=2, dropout=.1):
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids, hidden=None, return_hidden=False):
        # Right padded tokens never affect previous valid states. Loss ignores PAD.
        out, hidden = self.lstm(self.dropout(self.embedding(ids)), hidden)
        logits = output_logits(self.head(self.dropout(out)))
        return (logits, hidden) if return_hidden else logits


def build_model(config, vocab_size):
    cls = {"gpt": GPT, "lstm": LSTMLM}[config["model"]]
    return cls(vocab_size=vocab_size, **config["architecture"])


def parameter_count(model):
    # parameters() de-duplicates shared embedding/head tensors.
    return sum(p.numel() for p in model.parameters())
