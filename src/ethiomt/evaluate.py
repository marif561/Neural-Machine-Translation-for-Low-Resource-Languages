"""BLEU, TER and chrF++ (SacreBLEU) on each evaluation language for a trained run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sacrebleu

from .data import has_split, lang_tag, read_pair


def bleu(hyps: List[str], refs: List[str]) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def scores(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    return {"bleu": bleu(hyps, refs),
            "ter": sacrebleu.corpus_ter(hyps, [refs]).score,
            "chrf++": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score}


def main(argv=None):
    import torch

    from .translate import load_run, translate_lines

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--langs", nargs="+", default=None, help="defaults to the run's eval languages")
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta = load_run(args.run, device)
    reverse = meta.get("reverse", False)
    results: Dict[str, Dict[str, float]] = {}
    for lang in args.langs or meta["eval_langs"]:
        if not has_split(args.data_dir, lang, args.split):
            print(f"{lang}: no {args.split} split, skipped")
            continue
        pairs = read_pair(args.data_dir, lang, args.split, reverse)
        hyps = translate_lines(model, tok, [s for s, _ in pairs], meta["max_len"], device,
                               tag=lang_tag(lang) if meta["tags"] else None)
        results[lang] = scores(hyps, [t for _, t in pairs])
        (Path(args.run) / f"{args.split}.{lang}.hyp").write_text("\n".join(hyps) + "\n", encoding="utf-8")
        direction = f"en -> {lang}" if reverse else f"{lang} -> en"
        r = results[lang]
        print(f"{direction:>14}  BLEU {r['bleu']:5.2f}  TER {r['ter']:6.2f}  chrF++ {r['chrf++']:5.2f}")
    (Path(args.run) / f"{args.split}_scores.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
