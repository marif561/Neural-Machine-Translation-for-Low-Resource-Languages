from pathlib import Path

from ethiomt.data import load_corpus, read_pair
from ethiomt.evaluate import bleu
from ethiomt.tokenizer import BOS, EOS, train_bpe

TOY = Path(__file__).resolve().parents[1] / "data" / "toy"
LANGS = ["toy_a", "toy_b", "toy_c"]


def test_read_pair_aligned():
    pairs = read_pair(TOY, "toy_a", "train")
    assert len(pairs) == 60 and all(s and t for s, t in pairs)


def test_mixed_training_pools_train_keeps_eval_separate():
    single = load_corpus(TOY, ["toy_a"])
    mixed = load_corpus(TOY, LANGS, tags=True)
    assert len(mixed.train) == 3 * len(single.train)
    assert set(mixed.dev) == set(LANGS) and len(mixed.test["toy_b"]) == 10
    assert all(s.split()[0] in {"<toy_a>", "<toy_b>", "<toy_c>"} for s, _ in mixed.train)
    assert mixed.test["toy_c"][0][0].startswith("<toy_c> ")


def test_bpe_keeps_tags_as_single_tokens(tmp_path):
    corpus = load_corpus(TOY, LANGS, tags=True)
    tok = train_bpe([s for s, _ in corpus.train] + [t for _, t in corpus.train], tmp_path / "spm", 120,
                    ["<toy_a>", "<toy_b>", "<toy_c>"])
    ids = tok.encode("<toy_b> ehta remrafa", max_len=32)
    assert ids[0] == BOS and ids[-1] == EOS
    pieces = [tok.sp.id_to_piece(i) for i in ids[1:4]]
    assert "<toy_b>" in pieces  # the tag is one token, not split into sub-words
    assert tok.decode(tok.encode("the farmer", 32)) == "the farmer"


def test_bleu():
    assert abs(bleu(["the farmer finds a friend"], ["the farmer finds a friend"]) - 100.0) < 1e-6
