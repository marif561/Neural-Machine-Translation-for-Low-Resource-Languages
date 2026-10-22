"""Model and end-to-end checks (skipped when PyTorch is not installed)."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ethiomt.evaluate import main as eval_main  # noqa: E402
from ethiomt.model import Seq2SeqTransformer  # noqa: E402
from ethiomt.train import main as train_main  # noqa: E402
from ethiomt.translate import main as translate_main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "data" / "toy"


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


def test_mixed_training_end_to_end(tmp_path, capsys):
    run = tmp_path / "run"
    train_main(["--data-dir", str(TOY), "--langs", "toy_a", "toy_b", "toy_c", "--tags",
                "--config", str(ROOT / "configs/tiny.yaml"), "--out", str(run)])
    assert (run / "model.pt").exists() and json.loads((run / "meta.json").read_text())["tags"]
    eval_main(["--run", str(run), "--data-dir", str(TOY), "--langs", "toy_a", "toy_b"])
    scores = json.loads((run / "test_bleu.json").read_text())
    assert set(scores) == {"toy_a", "toy_b"}
    translate_main(["--run", str(run), "--lang", "toy_c", "--input", str(TOY / "toy_c/dev.src"),
                    "--output", str(tmp_path / "hyp.txt")])
    assert len((tmp_path / "hyp.txt").read_text().splitlines()) == 10
