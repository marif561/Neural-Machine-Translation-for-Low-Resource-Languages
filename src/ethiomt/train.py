"""Train a Transformer NMT model on one language pair (baseline) or on several related
pairs at once (mixed training).

    --langs gamo                      single-pair baseline
    --langs gamo gofa dawuro --tags   mixed training with source-language tags

Optimisation follows Vaswani et al. (2017): Adam (0.9, 0.98), inverse-square-root
warm-up schedule, label smoothing 0.1. The checkpoint with the best average dev BLEU
across the trained pairs is kept.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import lang_tag, load_corpus
from .evaluate import bleu
from .model import Seq2SeqTransformer
from .tokenizer import PAD, train_bpe
from .translate import translate_lines


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(pairs, tok, max_len, max_tokens, seed):
    """Length-bucketed batches of (src, tgt) id tensors."""
    enc = [(tok.encode(s, max_len), tok.encode(t, max_len)) for s, t in pairs]
    enc.sort(key=lambda p: (len(p[0]), len(p[1])))
    batches, cur, width = [], [], 0
    for p in enc:
        width = max(width, len(p[0]), len(p[1]))
        if cur and width * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, width = [], max(len(p[0]), len(p[1]))
        cur.append(p)
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)

    def pad(seqs):
        w = max(map(len, seqs))
        return torch.tensor([s + [PAD] * (w - len(s)) for s in seqs])
    return [(pad([s for s, _ in b]), pad([t for _, t in b])) for b in batches]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--tags", action="store_true", help="prefix each source sentence with <lang>")
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    m, t = cfg["model"], cfg["train"]
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.data_dir, args.langs, args.tags, args.seed)
    print(f"training pairs: {len(corpus.train)} from {args.langs} (tags={args.tags})")
    tags = [lang_tag(l) for l in args.langs] if args.tags else []
    tok = train_bpe([s for s, _ in corpus.train] + [x for _, x in corpus.train], out / "spm",
                    t["vocab_size"], tags)
    batches = make_batches(corpus.train, tok, t["max_len"], t["max_tokens"], args.seed)

    model = Seq2SeqTransformer(len(tok), m["d_model"], m["heads"], m["layers"], m["ff_dim"], m["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    warmup, factor = t["warmup_steps"], float(t["lr_factor"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: factor * m["d_model"] ** -0.5 * min((s + 1) ** -0.5, (s + 1) * warmup ** -1.5))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=t["label_smoothing"])

    meta = {"langs": args.langs, "tags": args.tags, "max_len": t["max_len"], "model": m}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    best, bad = -1.0, 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        total, ntok = 0.0, 0
        for src, tgt in batches:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["clip_norm"])
            opt.step()
            sched.step()
            n = tgt[:, 1:].ne(PAD).sum().item()
            total, ntok = total + loss.item() * n, ntok + n

        model.eval()
        dev_scores = {}
        for lang, pairs in corpus.dev.items():
            hyps = translate_lines(model, tok, [s for s, _ in pairs], t["max_len"], device)  # tags already added
            dev_scores[lang] = bleu(hyps, [x for _, x in pairs])
        avg = sum(dev_scores.values()) / max(len(dev_scores), 1)
        print(f"epoch {epoch:>3}  loss {total / max(ntok, 1):.4f}  dev BLEU "
              + "  ".join(f"{l} {s:.2f}" for l, s in dev_scores.items()))
        if avg > best or not dev_scores:
            best, bad = avg, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad += 1
            if bad >= t["patience"]:
                print(f"early stopping (patience {t['patience']})")
                break
    print(f"best average dev BLEU {best:.2f} · run saved to {out}/")


if __name__ == "__main__":
    main()
