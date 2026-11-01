# Movie Review Sentiment Classifier

AI course project — Collegium Da Vinci, Computer Science, group L3, academic year 2026/2027.

**Problem:** Editors of a film review site need to know whether a new user review is positive or negative without reading every one of them.
**AI technique:** supervised text classification (classic machine learning / neural network) — to be confirmed in the topic card.
**Baseline:** always predicting the most frequent class; a simple keyword-list classifier.

> Status: project skeleton. The repository is maintained in English. Two exceptions required by the course: the topic card (`karta-tematu.md`, the course submission form) and the full project documentation (`docs/dokumentacja-pl.md`) are in Polish.

## Team

- Adrianna Golak
- Gilbert Helwich
- Dawid Ślusarczyk

## Repository structure

```
src/        core implementation (Python modules)
notebooks/  experiments (Google Colab)
data/       dataset description and loading scripts
tests/      tests
results/    plots, confusion matrices, comparison with the baseline
docs/       Polish documentation (8–15 pages) and English technical summary
```

## How to run

Requirements: Python 3.10+.

```
pip install -r requirements.txt
```

### Data

The project uses the [Filmweb+](https://github.com/narolski/filmwebplus) dataset (Paweł Narolski, 2020, MIT license). Put `filmwebplus.gz` in `data/raw/filmwebplus/`. This folder is not committed.

### Train/test split

```
python -m src.split
```

This keeps only negative and positive reviews, removes duplicates, balances the two classes and splits them 80/20 with a fixed seed. Only review ids and labels are saved, to `data/split/train_ids.csv` and `data/split/test_ids.csv`. A review id is the row number in the raw file. These files are committed and **must not change**: every model uses exactly the same split. Running the script again gives the same files (a test checks this); do not edit them by hand or change the seed. In code, load the data with `src.split.load_split()`.

Current split: 4,936 training reviews and 1,234 test reviews, 50% positive in each part.

### Tests

```
python -m pytest
```

Tests that need the raw data are skipped if `data/raw/filmwebplus/filmwebplus.gz` is missing.

## Use of AI tools

_TBD._
