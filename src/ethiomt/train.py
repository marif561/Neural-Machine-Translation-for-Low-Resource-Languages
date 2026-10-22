"""Train a Transformer NMT model for the paper's experiments.

    baseline        --train-langs gamo
    Exp 1 (TL)      parent:  --train-langs wolaita
                    child:   --train-langs gamo --init-from runs/exp1_parent
    Exp 2           --support-langs wolaita --train-langs gamo
    Exp 3 (0-shot)  --support-langs wolaita --eval-langs gamo gofa dawuro   (with the zero-shot splits)
    Exp 4           --support-langs wolaita gofa dawuro --train-langs gamo
    English -> X    add --reverse

Settings follow the paper (Table 2): 6 encoder and 6 decoder layers, model size 512,
Adam with learning rate 1e-4, dropout 0.3, token-based batches of 3,072 tokens, BPE
subwords. The checkpoint with the best average dev BLEU is kept.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import lang_tag, load_corpus
from .evaluate import bleu
from .model import Seq2SeqTransformer
from .tokenizer import PAD, Tokenizer, train_bpe
from .translate import translate_lines


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(pairs, tok, max_len, max_tokens, seed):
    """Length-bucketed batches of (src, tgt) id tensors with about ``max_tokens`` tokens each."""
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


def make_scheduler(opt, t, d_model):
    if t.get("schedule", "constant") == "noam":
        warmup, factor = t["warmup_steps"], float(t["lr_factor"])
        for g in opt.param_groups:
            g["lr"] = 1.0
        return torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: factor * d_model ** -0.5 * min((s + 1) ** -0.5, (s + 1) * warmup ** -1.5))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-langs", nargs="*", default=[], help="languages whose training split is used")
    ap.add_argument("--support-langs", nargs="*", default=[], help="languages whose data is used in full")
    ap.add_argument("--eval-langs", nargs="*", default=None, help="dev/test languages (default: --train-langs)")
    ap.add_argument("--reverse", action="store_true", help="translate English -> Ethiopian language")
    ap.add_argument("--tags", action="store_true", help="prefix sentences with a <lang> tag (not used in the paper)")
    ap.add_argument("--init-from", default=None, help="parent run to fine-tune from (Exp 1, transfer learning)")
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
    eval_langs = args.eval_langs if args.eval_langs is not None else (args.train_langs or args.support_langs)

    corpus = load_corpus(args.data_dir, args.train_langs, args.support_langs, eval_langs, args.reverse,
                         args.tags, args.seed)
    print(f"training pairs: {len(corpus.train):,}  support={args.support_langs}  train={args.train_langs}  "
          f"eval={eval_langs}  direction={'en->X' if args.reverse else 'X->en'}")

    if args.init_from:  # transfer learning: reuse the parent's vocabulary, architecture and weights
        parent = Path(args.init_from)
        shutil.copy(parent / "spm.model", out / "spm.model")
        tok = Tokenizer(out / "spm.model")
        m = json.loads((parent / "meta.json").read_text())["model"]
    else:
        tags = [lang_tag(l) for l in set(args.train_langs) | set(args.support_langs) | set(eval_langs)] if args.tags else []
        tok = train_bpe([s for s, _ in corpus.train] + [x for _, x in corpus.train], out / "spm", t["vocab_size"], tags)
    batches = make_batches(corpus.train, tok, t["max_len"], t["max_tokens"], args.seed)

    model = Seq2SeqTransformer(len(tok), m["d_model"], m["heads"], m["layers"], m["ff_dim"], m["dropout"]).to(device)
    if args.init_from:
        model.load_state_dict(torch.load(Path(args.init_from) / "model.pt", map_location=device))
    opt = torch.optim.Adam(model.parameters(), lr=float(t["learning_rate"]), betas=(0.9, 0.98), eps=1e-9)
    sched = make_scheduler(opt, t, m["d_model"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=t["label_smoothing"])

    meta = {"train_langs": args.train_langs, "support_langs": args.support_langs, "eval_langs": eval_langs,
            "reverse": args.reverse, "tags": args.tags, "init_from": args.init_from,
            "max_len": t["max_len"], "model": m}
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
            if sched:
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
