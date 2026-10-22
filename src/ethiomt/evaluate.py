"""SacreBLEU on each language pair's test set for a trained run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sacrebleu

from .data import read_pair


def bleu(hyps: List[str], refs: List[str]) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def main(argv=None):
    import torch

    from .data import lang_tag
    from .translate import load_run, translate_lines

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta = load_run(args.run, device)
    scores: Dict[str, float] = {}
    for lang in args.langs:
        pairs = read_pair(args.data_dir, lang, args.split)
        hyps = translate_lines(model, tok, [s for s, _ in pairs], meta["max_len"], device,
                               tag=lang_tag(lang) if meta["tags"] else None)
        scores[lang] = bleu(hyps, [t for _, t in pairs])
        (Path(args.run) / f"{args.split}.{lang}.hyp").write_text("\n".join(hyps) + "\n", encoding="utf-8")
        print(f"{lang:>8} -> en  BLEU {scores[lang]:.2f}")
    (Path(args.run) / f"{args.split}_bleu.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
