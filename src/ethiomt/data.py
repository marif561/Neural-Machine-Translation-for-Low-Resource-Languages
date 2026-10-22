"""Parallel corpus loading and mixed training.

Each language pair lives in ``<data_dir>/<lang>/{train,dev,test}.{src,tgt}``: ``.src`` is the
Ethiopian-language side and ``.tgt`` is English, one sentence per line. A language may
have only some of the splits (e.g. Wolaita with just ``train`` and ``dev``).

Mixed training (paper Sect. 4.2) builds one training set from
  * **support languages**: all of their data (every split that exists), and
  * **train languages**: their training split only,
and evaluates on the dev/test splits of the **eval languages**.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Pair = Tuple[str, str]
SPLITS = ("train", "dev", "test")


def lang_tag(lang: str) -> str:
    return f"<{lang}>"


def has_split(data_dir: str | Path, lang: str, split: str) -> bool:
    return (Path(data_dir) / lang / f"{split}.src").exists()


def read_pair(data_dir: str | Path, lang: str, split: str, reverse: bool = False) -> List[Pair]:
    """Read one split. With ``reverse`` the pairs become English -> Ethiopian language."""
    d = Path(data_dir) / lang
    src = (d / f"{split}.src").read_text(encoding="utf-8").splitlines()
    tgt = (d / f"{split}.tgt").read_text(encoding="utf-8").splitlines()
    if len(src) != len(tgt):
        raise ValueError(f"{d}/{split}: {len(src)} source vs {len(tgt)} target lines")
    pairs = [(s.strip(), t.strip()) for s, t in zip(src, tgt) if s.strip() and t.strip()]
    return [(t, s) for s, t in pairs] if reverse else pairs


def add_tag(pairs: Sequence[Pair], tag: str) -> List[Pair]:
    return [(f"{tag} {s}", t) for s, t in pairs]


@dataclass
class Corpus:
    train: List[Pair]
    dev: Dict[str, List[Pair]]
    test: Dict[str, List[Pair]]


def load_corpus(data_dir: str | Path, train_langs: Sequence[str] = (), support_langs: Sequence[str] = (),
                eval_langs: Sequence[str] | None = None, reverse: bool = False, tags: bool = False,
                seed: int = 42) -> Corpus:
    """Build a (possibly mixed) training set and per-language dev/test sets.

    ``tags`` prefixes each source sentence with ``<lang>`` (only meaningful for X -> English;
    for English -> X the tag names the *target* language instead).
    """
    eval_langs = list(eval_langs) if eval_langs is not None else list(train_langs)

    def prep(lang, split):
        pairs = read_pair(data_dir, lang, split, reverse)
        return add_tag(pairs, lang_tag(lang)) if tags else pairs

    train: List[Pair] = []
    for lang in support_langs:
        for split in SPLITS:
            if has_split(data_dir, lang, split):
                train += prep(lang, split)
    for lang in train_langs:
        train += prep(lang, "train")
    if not train:
        raise ValueError("empty training set: give --train-langs and/or --support-langs")
    random.Random(seed).shuffle(train)
    dev = {l: prep(l, "dev") for l in eval_langs if has_split(data_dir, l, "dev")}
    test = {l: prep(l, "test") for l in eval_langs if has_split(data_dir, l, "test")}
    return Corpus(train, dev, test)
