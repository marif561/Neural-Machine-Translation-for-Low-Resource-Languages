"""Parallel corpus loading and mixing.

Each language pair lives in ``<data_dir>/<lang>/{train,dev,test}.{src,tgt}`` with one
sentence per line. Mixed training pools the training data of several related source
languages into one corpus; an optional tag such as ``<gamo>`` marks the source language.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Pair = Tuple[str, str]


def lang_tag(lang: str) -> str:
    return f"<{lang}>"


def read_pair(data_dir: str | Path, lang: str, split: str) -> List[Pair]:
    d = Path(data_dir) / lang
    src = (d / f"{split}.src").read_text(encoding="utf-8").splitlines()
    tgt = (d / f"{split}.tgt").read_text(encoding="utf-8").splitlines()
    if len(src) != len(tgt):
        raise ValueError(f"{d}/{split}: {len(src)} source vs {len(tgt)} target lines")
    return [(s.strip(), t.strip()) for s, t in zip(src, tgt) if s.strip() and t.strip()]


def add_tag(pairs: Sequence[Pair], lang: str) -> List[Pair]:
    return [(f"{lang_tag(lang)} {s}", t) for s, t in pairs]


@dataclass
class Corpus:
    train: List[Pair]
    dev: Dict[str, List[Pair]]
    test: Dict[str, List[Pair]]


def load_corpus(data_dir: str | Path, langs: Sequence[str], tags: bool = False, seed: int = 42) -> Corpus:
    """Load one or more language pairs. With several ``langs`` the training data is mixed.

    Dev and test sets stay separate per language so each pair is evaluated on its own.
    """
    def prep(lang, split):
        pairs = read_pair(data_dir, lang, split)
        return add_tag(pairs, lang) if tags else pairs

    train: List[Pair] = []
    for lang in langs:
        train += prep(lang, "train")
    random.Random(seed).shuffle(train)
    dev = {l: prep(l, "dev") for l in langs if (Path(data_dir) / l / "dev.src").exists()}
    test = {l: prep(l, "test") for l in langs if (Path(data_dir) / l / "test.src").exists()}
    return Corpus(train, dev, test)
