"""Model and end-to-end experiment checks (skipped when PyTorch is not installed)."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ethiomt.evaluate import main as eval_main  # noqa: E402
from ethiomt.model import Seq2SeqTransformer  # noqa: E402
from ethiomt.train import main as train_main  # noqa: E402
from ethiomt.translate import main as translate_main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOY = str(ROOT / "data" / "toy")
TINY = str(ROOT / "configs" / "tiny.yaml")


def test_forward_and_greedy():
    model = Seq2SeqTransformer(50, d_model=32, heads=4, layers=2, ff_dim=64, dropout=0.1)
    src = torch.randint(4, 50, (3, 7))
    src[0, 5:] = 0
    tgt = torch.randint(4, 50, (3, 6))
    logits = model(src, tgt)
    assert logits.shape == (3, 6, 50)
    logits.sum().backward()
    out = model.eval().greedy(src, max_len=10)
    assert out.shape[0] == 3 and out.shape[1] <= 10


def test_exp4_mixed_training(tmp_path):
    run = tmp_path / "exp4"
    train_main(["--data-dir", TOY, "--support-langs", "toy_w", "toy_b", "toy_c", "--train-langs", "toy_a",
                "--config", TINY, "--out", str(run)])
    eval_main(["--run", str(run), "--data-dir", TOY])
    res = json.loads((run / "test_scores.json").read_text())
    assert set(res) == {"toy_a"} and set(res["toy_a"]) == {"bleu", "ter", "chrf++"}


def test_exp1_transfer_learning_and_reverse(tmp_path):
    parent, child = tmp_path / "parent", tmp_path / "child"
    train_main(["--data-dir", TOY, "--train-langs", "toy_w", "--config", TINY, "--out", str(parent)])
    train_main(["--data-dir", TOY, "--train-langs", "toy_a", "--init-from", str(parent),
                "--config", TINY, "--out", str(child)])
    assert (child / "spm.model").read_bytes() == (parent / "spm.model").read_bytes()
    assert json.loads((child / "meta.json").read_text())["init_from"] == str(parent)

    rev = tmp_path / "reverse"
    train_main(["--data-dir", TOY, "--support-langs", "toy_w", "--train-langs", "toy_b", "--reverse",
                "--config", TINY, "--out", str(rev)])
    eval_main(["--run", str(rev), "--data-dir", TOY])
    translate_main(["--run", str(rev), "--lang", "toy_b", "--input", str(Path(TOY) / "toy_b/dev.tgt"),
                    "--output", str(tmp_path / "hyp.txt")])
    assert len((tmp_path / "hyp.txt").read_text().splitlines()) == 8
