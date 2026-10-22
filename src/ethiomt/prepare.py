"""Clean and split a raw parallel corpus (paper Sect. 4.1 and 5.2).

Cleaning: lower-case, remove digits and characters outside the Latin script, collapse
spaces, and drop empty and duplicate sentence pairs. The cleaned pairs are shuffled and
split by the given ratios into train / dev / test.

    # main experiments (about 64 / 16 / 20, as in Table 3)
    python -m ethiomt.prepare --src raw/gamo.txt --tgt raw/gamo.en.txt --out data/main/gamo --ratios 0.64 0.16 0.20
    # zero-shot experiment 3 (dev / test only, about 20 / 80)
    python -m ethiomt.prepare --src raw/gamo.txt --tgt raw/gamo.en.txt --out data/zeroshot/gamo --ratios 0 0.2 0.8
"""
from __future__ import annotations

import argparse
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Sequence, Tuple

DIGITS_RE = re.compile(r"\d+")
NON_LATIN_RE = re.compile(r"[^a-z\s'\-.,;:!?]")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))  # strip accents, keep base letter
    text = DIGITS_RE.sub(" ", text)
    text = NON_LATIN_RE.sub(" ", text)
    # drop punctuation left on its own, e.g. the ":" of a removed verse reference "3:16"
    return " ".join(tok for tok in text.split() if any(c.isalpha() for c in tok))


def clean_pairs(src: Sequence[str], tgt: Sequence[str]) -> List[Tuple[str, str]]:
    if len(src) != len(tgt):
        raise ValueError(f"{len(src)} source vs {len(tgt)} target lines")
    seen, out = set(), []
    for s, t in zip(src, tgt):
        s, t = clean(s), clean(t)
        if s and t and (s, t) not in seen:
            seen.add((s, t))
            out.append((s, t))
    return out


def split_pairs(pairs, ratios: Sequence[float], seed: int = 42):
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    a = round(ratios[0] * n)
    b = a + round(ratios[1] * n)
    return {"train": pairs[:a], "dev": pairs[a:b], "test": pairs[b:]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Ethiopian-language file, one sentence per line")
    ap.add_argument("--tgt", required=True, help="aligned English file")
    ap.add_argument("--out", required=True, help="output folder, e.g. data/main/gamo")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.64, 0.16, 0.20], metavar=("TRAIN", "DEV", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    src = Path(args.src).read_text(encoding="utf-8").splitlines()
    tgt = Path(args.tgt).read_text(encoding="utf-8").splitlines()
    pairs = clean_pairs(src, tgt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, part in split_pairs(pairs, args.ratios, args.seed).items():
        if not part:
            continue
        (out / f"{split}.src").write_text("\n".join(s for s, _ in part) + "\n", encoding="utf-8")
        (out / f"{split}.tgt").write_text("\n".join(t for _, t in part) + "\n", encoding="utf-8")
        print(f"{split:>5}: {len(part):,} pairs")
    print(f"kept {len(pairs):,} of {len(src):,} pairs after cleaning and de-duplication")


if __name__ == "__main__":
    main()
