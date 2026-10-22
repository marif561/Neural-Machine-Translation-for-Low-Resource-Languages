"""Transformer encoder–decoder (Vaswani et al., 2017) with shared embeddings."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .tokenizer import BOS, EOS, PAD


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, heads: int = 8, layers: int = 6,
                 ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)  # shared src/tgt vocabulary
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, heads, layers, layers, ff_dim, dropout, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size)
        self.out.weight = self.embed.weight  # tie output projection to embeddings

    def _emb(self, x):
        return self.drop(self.pos(self.embed(x) * math.sqrt(self.d_model)))

    def encode(self, src):
        return self.transformer.encoder(self._emb(src), src_key_padding_mask=src.eq(PAD))

    def decode(self, tgt, memory, src_pad):
        causal = self.transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        h = self.transformer.decoder(self._emb(tgt), memory, tgt_mask=causal,
                                     tgt_key_padding_mask=tgt.eq(PAD), memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src), src.eq(PAD))

    @torch.no_grad()
    def greedy(self, src, max_len: int = 128):
        memory, src_pad = self.encode(src), src.eq(PAD)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            nxt = self.decode(ys, memory, src_pad)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
            done |= nxt.eq(EOS)
            if done.all():
                break
        return ys
