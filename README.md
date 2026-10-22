<div align="center">

# Neural Machine Translation for Low-Resource Ethiopian Languages
### Mixed training with related languages: Gamo, Gofa and Dawuro ↔ English

[![Paper](https://img.shields.io/badge/MICAI%202022-LNAI%2013613-1f6feb)](https://doi.org/10.1007/978-3-031-19496-2_3)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--19496--2__3-blue)](https://doi.org/10.1007/978-3-031-19496-2_3)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB)](#installation)
[![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![tests](https://github.com/marif561/Neural-Machine-Translation-for-Low-Resource-Languages/actions/workflows/tests.yml/badge.svg)](https://github.com/marif561/Neural-Machine-Translation-for-Low-Resource-Languages/actions/workflows/tests.yml)

Code and results for **"Improving Neural Machine Translation for Low Resource Languages Using Mixed Training: The Case of Ethiopian Languages"**, MICAI 2022, *Advances in Computational Intelligence*, LNAI 13613, pp. 30–40, Springer.

Atnafu Lambebo Tonja · Olga Kolesnikova · **Muhammad Arif** · Alexander Gelbukh · Grigori Sidorov

*Centro de Investigación en Computación, Instituto Politécnico Nacional (CIC-IPN), Mexico City*

</div>

---

## About the paper

Neural machine translation (NMT) works well when millions of parallel sentences exist, but many languages have only a few thousand, often from a single domain. Earlier work improved low-resource NMT with transfer learning from a *high-resource* parent language such as French. This paper instead improves translation **without any high-resource language** by mixing data from **closely related low-resource languages**. It studies four Omotic languages of southern Ethiopia: **Wolaita**, **Gamo**, **Gofa** and **Dawuro**. Wolaita has the most data (26,943 sentence pairs) and the other three about 7,800 each. When Wolaita and the other related languages are pooled into one Transformer training set, **X → English** translation improves over previous work on the same data by **+5.3 BLEU for Gamo**, **+5.0 for Gofa** and **+3.7 for Dawuro**.

<p align="center">
  <img src="docs/figures/mixed_training.svg" alt="Mixed training for related low-resource languages" width="860">
</p>

## Data

Bible-domain parallel corpora from Tonja et al. (2021):

| Pair | Sentences | Tokens (Ethiopian / English) | Types (Ethiopian / English) | Avg. words per sentence (Ethiopian / English) |
|---|---:|---:|---:|:-:|
| Wolaita – English | 26,943 | 469,851 / 703,122 | 42,049 / 12,131 | 17 / 26 |
| Gamo – English | 7,866 | 125,509 / 177,410 | 23,589 / 11,078 | 16 / 23 |
| Gofa – English | 7,928 | 119,289 / 175,727 | 25,301 / 8,769 | 15 / 22 |
| Dawuro – English | 7,804 | 126,734 / 207,954 | 17,392 / 4,368 | 16 / 27 |

The Ethiopian languages have many more word types than English for the same text, which reflects their rich morphology. Preprocessing removes duplicates, digits and non-Latin characters and lower-cases everything. Text is split into **BPE** subwords with a vocabulary shared across languages.

## Experiments

| Exp | Training data | Evaluated on |
|---|---|---|
| **1** · Transfer learning | Train a Wolaita→English *parent*, then fine-tune it on each pair | Gamo / Gofa / Dawuro test |
| **2** · Mixed (2 languages) | All Wolaita data + the target language's training split | Target test |
| **3** · Zero-shot | All Wolaita data only | Unseen Gamo / Gofa / Dawuro |
| **4** · Mixed (4 languages) | All Wolaita data + the other two languages + the target's training split | Target test |

Model (Table 2): a Transformer with **6 encoder and 6 decoder layers**, model size **512**, **Adam** with learning rate **1e-4**, dropout **0.3**, and **3,072-token** batches, trained with OpenNMT-tf on Google Colab Pro+.

## Results

### X → English (Table 4)

| Pair | Exp 1 BLEU / TER / chrF++ | Exp 2 | Exp 3 | **Exp 4** |
|---|:-:|:-:|:-:|:-:|
| Gamo → En | 1.4 / 106.2 / 20.4 | 6.5 / 88.0 / 22.8 | 4.8 / 94.4 / 22.2 | **9.4 / 83.6 / 26.9** |
| Gofa → En | 5.6 / 99.3 / 22.4 | 7.2 / 86.8 / 25.1 | 5.4 / 93.1 / 25.6 | **9.5 / 83.3 / 27.1** |
| Dawuro → En | 3.9 / 92.5 / 21.5 | 6.2 / 87.2 / 23.1 | 2.2 / 101.7 / 20.9 | **7.3 / 83.5 / 26.5** |

Pooling more related languages gives the best result for every pair. Higher BLEU and chrF++ are better; lower TER is better.

### English → X (Table 5)

| Pair | Exp 2 BLEU / TER / chrF++ | Exp 4 |
|---|:-:|:-:|
| En → Gamo | **2.0 / 100.7 / 21.0** | 1.5 / 103.2 / 20.7 |
| En → Gofa | **2.9 / 97.5 / 22.3** | 2.1 / 100.5 / 21.2 |
| En → Dawuro | **2.6 / 97.7 / 21.4** | 1.8 / 101.4 / 21.2 |

Translating *into* the morphologically rich Ethiopian languages is much harder. Here the two-language mix (Exp 2) works better than the four-language mix.

### Comparison with previous work on the same data (Table 6, BLEU)

| Method | Gamo En→X / X→En | Gofa | Dawuro |
|---|:-:|:-:|:-:|
| Tonja et al. (2021) | 2.2 / 4.1 | 2.4 / 4.5 | 2.1 / 3.6 |
| Yigezu et al. (2021) | – / 3.4 | – / 4.5 | – / 2.5 |
| **Proposed, Exp 2** | 2.0 / 6.5 | **2.9** / 7.2 | **2.6** / 6.2 |
| **Proposed, Exp 4** | 1.5 / **9.4** | 2.1 / **9.5** | 1.8 / **7.3** |

The headline gains (+5.3, +5.0, +3.7) are Exp 4 against Tonja et al. (2021) for X → English. All tables, including the dataset splits, are in [`results/`](results/).

**Takeaways**

- Pooling related low-resource languages beats transfer learning from a single parent, and needs no high-resource language.
- Training on one related language and testing on another (zero-shot, Exp 3) already gives usable BLEU for Gamo and Gofa. This shows how close the languages are.
- English → X stays hard, with BLEU under 3. Many English words must map to one rich Ethiopian word form.

## Repository structure

```
.
├── configs/transformer.yaml # Paper settings (Table 2)
├── configs/tiny.yaml        # Small model for smoke tests
├── data/                    # Data preparation guide + synthetic toy corpus
├── docs/figures/            # Mixed-training diagram
├── results/                 # Every table from the paper as CSV
├── scripts/run_experiments.sh  # Exp 1–4 end to end
├── tests/                   # Unit tests (CI on every push)
└── src/ethiomt/
    ├── prepare.py           # Paper cleaning + train/dev/test splitting
    ├── data.py              # Support / train / eval language mixing, reverse direction
    ├── tokenizer.py         # Shared SentencePiece BPE
    ├── model.py             # Transformer encoder–decoder (PyTorch)
    ├── train.py             # All experiments, incl. transfer learning (--init-from)
    ├── translate.py         # Greedy decoding
    └── evaluate.py          # BLEU, TER, chrF++ (SacreBLEU)
```

## Installation

```bash
git clone https://github.com/marif561/Neural-Machine-Translation-for-Low-Resource-Languages.git
cd Neural-Machine-Translation-for-Low-Resource-Languages
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Prepare the data as described in [`data/README.md`](data/README.md), then:

```bash
# Exp 4 for Gamo: all Wolaita, Gofa and Dawuro data + Gamo training split
python -m ethiomt.train --data-dir data/main --support-langs wolaita gofa dawuro --train-langs gamo --out runs/exp4_gamo
python -m ethiomt.evaluate --run runs/exp4_gamo --data-dir data/main

# Exp 2 for Gofa, English -> Gofa
python -m ethiomt.train --data-dir data/main --support-langs wolaita --train-langs gofa --reverse --out runs/exp2_gofa_en2x

# Exp 1: Wolaita parent, then fine-tune on Dawuro
python -m ethiomt.train --data-dir data/main --train-langs wolaita --out runs/exp1_parent
python -m ethiomt.train --data-dir data/main --train-langs dawuro --init-from runs/exp1_parent --out runs/exp1_dawuro

# Everything (add REVERSE=1 for English -> X)
bash scripts/run_experiments.sh data/main data/zeroshot

# Translate new sentences
python -m ethiomt.translate --run runs/exp4_gamo --lang gamo --input my_sentences.txt
```

To check the pipeline in about a minute on CPU, run `pytest`. It trains Exp 1, Exp 2 (English → X) and Exp 4 on the synthetic toy corpus in `data/toy` with `configs/tiny.yaml`.

> **Note.** The authors used OpenNMT-tf. This repository is a clean PyTorch re-implementation of the same setup. The settings that Table 2 doesn't list are marked in [`configs/transformer.yaml`](configs/transformer.yaml): attention heads, feed-forward size, BPE vocabulary size, label smoothing and early stopping. They use OpenNMT-tf defaults or common choices. Exact scores can differ with these choices, random seeds and splits.

## Citation

```bibtex
@inproceedings{tonja2022mixed,
  title     = {Improving Neural Machine Translation for Low Resource Languages Using Mixed Training:
               The Case of {Ethiopian} Languages},
  author    = {Tonja, Atnafu Lambebo and Kolesnikova, Olga and Arif, Muhammad and
               Gelbukh, Alexander and Sidorov, Grigori},
  booktitle = {Advances in Computational Intelligence. MICAI 2022},
  series    = {Lecture Notes in Computer Science (LNAI)},
  volume    = {13613},
  pages     = {30--40},
  publisher = {Springer, Cham},
  year      = {2022},
  doi       = {10.1007/978-3-031-19496-2_3}
}
```

## Acknowledgements

Partially supported by CONACYT grant A1S-47854, SIP-IPN grants 20220852, 20220859 and 20221627, the Deep Learning Platform for Language Technologies at INAOE, and the Microsoft Latin America PhD Award.

## License

Code is released under the [MIT License](LICENSE). The parallel corpora are subject to their own licences.
