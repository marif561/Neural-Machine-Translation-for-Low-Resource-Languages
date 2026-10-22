# Data

Parallel corpora are not redistributed here. Put each language pair in its own folder with line-aligned files (one sentence per line):

```
data/
├── gamo/    train.src  train.tgt  dev.src  dev.tgt  test.src  test.tgt
├── gofa/    ...
└── dawuro/  ...
```

`.src` is the Ethiopian-language side and `.tgt` is English. Dev and test files are optional but recommended. Without a dev set, the last epoch is kept.

## Sources

- The paper's references point to **Tonja et al. (2021)**, *A parallel corpora for bi-directional neural machine translation for low resourced Ethiopian languages*, for Gamo/Gofa/Dawuro–English data.
- **EthioMT** (Tonja et al., 2024, [LREC-COLING RAIL workshop](https://aclanthology.org/2024.rail-1.12)) is a newer, larger parallel corpus covering more Ethiopian languages.

Follow each corpus's licence.

## Toy corpus

[`toy/`](toy/) contains three **synthetic** "related languages" (`toy_a`, `toy_b`, `toy_c`). They are made by applying deterministic word-level rewrites, such as reversing letters and adding a suffix, to simple English sentences. They are **not** Gamo, Gofa or Dawuro. They exist only so the full pipeline, including mixed training, can be tested in about a minute on a CPU.
