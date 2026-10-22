"""Translate sentences with a trained run (greedy decoding)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from .data import lang_tag
from .model import Seq2SeqTransformer
from .tokenizer import PAD, Tokenizer


def load_run(run_dir: str | Path, device: str = "cpu"):
    run = Path(run_dir)
    meta = json.loads((run / "meta.json").read_text())
    tok = Tokenizer(run / "spm.model")
    m = meta["model"]
    model = Seq2SeqTransformer(len(tok), m["d_model"], m["heads"], m["layers"], m["ff_dim"], m["dropout"])
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    return model.to(device).eval(), tok, meta


def translate_lines(model, tok, lines: List[str], max_len: int, device: str = "cpu", bs: int = 64,
                    tag: str | None = None) -> List[str]:
    out = []
    for i in range(0, len(lines), bs):
        chunk = [f"{tag} {l}" if tag else l for l in lines[i:i + bs]]
        ids = [tok.encode(s, max_len) for s in chunk]
        width = max(map(len, ids))
        src = torch.tensor([x + [PAD] * (width - len(x)) for x in ids], device=device)
        for row in model.greedy(src, max_len).tolist():
            out.append(tok.decode(row))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lang", required=True, help="source language (adds its tag if the run used tags)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta = load_run(args.run, device)
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    hyps = translate_lines(model, tok, lines, meta["max_len"], device,
                           tag=lang_tag(args.lang) if meta["tags"] else None)
    text = "\n".join(hyps) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
