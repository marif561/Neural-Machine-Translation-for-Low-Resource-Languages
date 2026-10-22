from pathlib import Path

import pytest

from ethiomt.data import load_corpus, read_pair
from ethiomt.evaluate import bleu, scores
from ethiomt.prepare import clean, clean_pairs, split_pairs
from ethiomt.tokenizer import BOS, EOS, train_bpe

TOY = Path(__file__).resolve().parents[1] / "data" / "toy"


def test_read_pair_and_reverse():
    fwd = read_pair(TOY, "toy_a", "train")
    rev = read_pair(TOY, "toy_a", "train", reverse=True)
    assert len(fwd) == 24 and fwd[0] == (rev[0][1], rev[0][0])


def test_mixed_training_like_exp4():
    # support language contributes all splits; target contributes train only; eval on target
    c = load_corpus(TOY, train_langs=["toy_a"], support_langs=["toy_w", "toy_b"], eval_langs=["toy_a"])
    assert len(c.train) == 24 + (100 + 20) + (24 + 8 + 8)
    assert set(c.dev) == {"toy_a"} and set(c.test) == {"toy_a"}


def test_zero_shot_like_exp3():
    c = load_corpus(TOY, support_langs=["toy_w"], eval_langs=["toy_a", "toy_b", "toy_c"])
    assert len(c.train) == 120 and set(c.test) == {"toy_a", "toy_b", "toy_c"}
    train_src = {s for s, _ in c.train}
    assert not train_src & {s for s, _ in c.test["toy_a"]}


def test_empty_training_set_is_an_error():
    with pytest.raises(ValueError):
        load_corpus(TOY)


def test_tags():
    c = load_corpus(TOY, train_langs=["toy_a", "toy_b"], tags=True)
    assert {s.split()[0] for s, _ in c.train} == {"<toy_a>", "<toy_b>"}


def test_paper_cleaning_and_split():
    assert clean("Yesus 3:16 KAALLIDI, ጤና  yesus!") == "yesus kaallidi, yesus!"
    pairs = clean_pairs(["A 1", "a 1", "B", ""], ["x", "x", "y", "z"])
    assert pairs == [("a", "x"), ("b", "y")]  # duplicates and empties dropped
    parts = split_pairs([(str(i), str(i)) for i in range(100)], [0.64, 0.16, 0.20])
    assert [len(parts[s]) for s in ("train", "dev", "test")] == [64, 16, 20]
    assert len(split_pairs([("a", "b")] * 10, [0, 0.2, 0.8])["train"]) == 0


def test_bpe_keeps_tags_as_single_tokens(tmp_path):
    c = load_corpus(TOY, train_langs=["toy_a", "toy_b"], tags=True)
    tok = train_bpe([s for s, _ in c.train] + [t for _, t in c.train], tmp_path / "spm", 120, ["<toy_a>", "<toy_b>"])
    ids = tok.encode("<toy_b> ehta remrafa", max_len=32)
    assert ids[0] == BOS and ids[-1] == EOS
    assert "<toy_b>" in [tok.sp.id_to_piece(i) for i in ids[1:4]]
    assert tok.decode(tok.encode("the farmer", 32)) == "the farmer"


def test_metrics():
    ref = ["the farmer finds a friend"]
    assert abs(bleu(ref, ref) - 100.0) < 1e-6
    s = scores(ref, ref)
    assert s["ter"] == 0.0 and abs(s["chrf++"] - 100.0) < 1e-6
