# Data

## Source

The parallel corpora come from **Tonja et al. (2021)**, *A parallel corpora for bi-directional neural machine translation for low resourced Ethiopian languages*, ICT4DA 2021. They are religious-domain (Bible) texts for four related Omotic languages paired with English:

| Pair | Sentences | Ethiopian-side tokens | English tokens |
|---|---:|---:|---:|
| Wolaita – English | 26,943 | 469,851 | 703,122 |
| Gamo – English | 7,866 | 125,509 | 177,410 |
| Gofa – English | 7,928 | 119,289 | 175,727 |
| Dawuro – English | 7,804 | 126,734 | 207,954 |

Wolaita has more data because it includes both the Old and New Testaments, while the others have only the New Testament. A newer, larger multi-domain corpus is available in [EthioMT](https://aclanthology.org/2024.rail-1.12) (Tonja et al., 2024). The data is not redistributed here. Follow each corpus's licence.

## Preparing the folders

`ethiomt.prepare` applies the paper's cleaning step: lower-casing, removing digits and non-Latin characters, and dropping duplicates. It then splits the data:

```bash
# Main experiments (Exp 1, 2, 4): about 64 / 16 / 20 like Table 3
for L in gamo gofa dawuro; do
  python -m ethiomt.prepare --src raw/$L.txt --tgt raw/$L.en.txt --out data/main/$L --ratios 0.64 0.16 0.20
done
# Wolaita: train / dev only (21,555 / 5,388 in Exp 1; all of it is used for training in Exp 2-4)
python -m ethiomt.prepare --src raw/wolaita.txt --tgt raw/wolaita.en.txt --out data/main/wolaita --ratios 0.8 0.2 0

# Zero-shot experiment (Exp 3): dev / test only, about 20 / 80
for L in gamo gofa dawuro; do
  python -m ethiomt.prepare --src raw/$L.txt --tgt raw/$L.en.txt --out data/zeroshot/$L --ratios 0 0.2 0.8
done
cp -r data/main/wolaita data/zeroshot/
```

Each language folder then holds line-aligned `{train,dev,test}.{src,tgt}` files: `.src` is the Ethiopian language and `.tgt` is English. Use `--reverse` at training time for English → X.

## Toy corpus

[`toy/`](toy/) contains four **synthetic** "related languages". They are made by applying deterministic word-level rewrites, such as reversing letters and swapping vowels, to simple English sentences. `toy_w` plays the larger support language, like Wolaita, and `toy_a`/`toy_b`/`toy_c` play the low-resource ones. They are **not** real Ethiopian languages. They only exist so every experiment can be smoke-tested in about a minute on a CPU.
