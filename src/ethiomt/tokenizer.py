"""Shared SentencePiece BPE vocabulary for source and target."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import sentencepiece as spm

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def train_bpe(sentences: Sequence[str], model_prefix: str | Path, vocab_size: int,
              user_symbols: Sequence[str] = ()) -> "Tokenizer":
    """Train a joint BPE model. Language tags are kept as single tokens."""
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences), model_prefix=str(model_prefix), model_type="bpe",
        vocab_size=vocab_size, character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        user_defined_symbols=list(user_symbols), hard_vocab_limit=False, minloglevel=2)
    return Tokenizer(f"{model_prefix}.model")


class Tokenizer:
    def __init__(self, model_file: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, max_len: int) -> List[int]:
        return [BOS] + self.sp.encode(text)[: max_len - 2] + [EOS]

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode([i for i in ids if i not in (PAD, BOS, EOS)])
