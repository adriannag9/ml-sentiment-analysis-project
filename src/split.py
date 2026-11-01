"""Prepare a balanced, fixed train/test split of the Filmweb+ dataset.

Steps, in this order:
1. Load the raw CSV and give every review an id (its row number in the file).
2. Keep only negative and positive reviews and relabel them as 0 / 1.
3. Remove duplicate reviews, so the same text cannot end up in both parts.
4. Balance the classes: keep as many positive reviews as there are negative ones.
5. Split into a training part and a test part, separately for each class,
   so both parts keep the 50/50 class ratio.
6. Save only the review ids and labels to data/split/. These files are
   committed and must not change afterwards: every model is trained and
   evaluated on exactly the same reviews.

Run from the repository root:
    python -m src.split
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/filmwebplus/filmwebplus.gz")
SPLIT_DIR = Path("data/split")

# Labels in the raw data: -1 unlabelled, 0 negative, 1 neutral, 2 positive.
RAW_NEGATIVE = 0
RAW_POSITIVE = 2

# Labels used in the whole project.
NEGATIVE = 0
POSITIVE = 1

SEED = 42
TEST_SIZE = 0.2


def load_raw(path=RAW_PATH):
    """Read the raw dataset and return columns: review_id, text, raw_label.

    The file has two index columns left over from pandas, and they are not
    unique, so we use the row number in the file as the review id instead.
    """
    df = pd.read_csv(path, compression="gzip")
    return pd.DataFrame(
        {
            "review_id": np.arange(len(df)),
            "text": df["review"],
            "raw_label": df["sentiment"],
        }
    )


def normalize_for_dedup(text):
    """Text used only to detect duplicates: lowercase, single spaces.

    Two reviews that differ only in letter case or whitespace count as the
    same review. The original text is not changed.
    """
    return " ".join(str(text).lower().split())


def select_binary(df):
    """Keep negative and positive reviews and add a 0/1 `label` column."""
    keep = df["raw_label"].isin([RAW_NEGATIVE, RAW_POSITIVE])
    out = df[keep].copy()
    out["label"] = np.where(out["raw_label"] == RAW_POSITIVE, POSITIVE, NEGATIVE)
    return out


def remove_duplicates(df):
    """Remove repeated reviews.

    - If the same text appears with different labels, we cannot tell which
      label is right, so all its copies are removed.
    - If the same text appears more than once with the same label, the first
      copy (lowest review_id) is kept.
    """
    out = df.copy()
    out["_key"] = out["text"].map(normalize_for_dedup)

    labels_per_text = out.groupby("_key")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    out = out[~out["_key"].isin(conflicting)]

    out = out.sort_values("review_id").drop_duplicates(subset="_key", keep="first")
    return out.drop(columns="_key")


def balance_classes(df, seed=SEED):
    """Randomly drop reviews from the larger class until both are equal."""
    rng = np.random.default_rng(seed)
    n_per_class = df["label"].value_counts().min()

    parts = []
    for label in (NEGATIVE, POSITIVE):
        ids = df.loc[df["label"] == label, "review_id"].to_numpy()
        chosen = rng.choice(ids, size=n_per_class, replace=False)
        parts.append(df[df["review_id"].isin(chosen)])
    return pd.concat(parts).sort_values("review_id")


def split_train_test(df, test_size=TEST_SIZE, seed=SEED):
    """Split each class separately, so train and test keep the class ratio.

    For each class: shuffle the review ids with a fixed seed and put the
    first `test_size` share of them into the test part.
    """
    rng = np.random.default_rng(seed)
    test_ids = []
    for label in (NEGATIVE, POSITIVE):
        ids = df.loc[df["label"] == label, "review_id"].to_numpy()
        ids = rng.permutation(ids)
        n_test = round(len(ids) * test_size)
        test_ids.extend(ids[:n_test])

    is_test = df["review_id"].isin(test_ids)
    return df[~is_test], df[is_test]


def save_split(train, test, out_dir=SPLIT_DIR):
    """Save review ids and labels, sorted by id, one file per part."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("test", test)):
        part[["review_id", "label"]].sort_values("review_id").to_csv(
            out_dir / f"{name}_ids.csv", index=False
        )


def load_split(raw=None, split_dir=SPLIT_DIR):
    """Return (train, test) data frames with columns review_id, text, label.

    This is how the rest of the project should get the data: always through
    the saved split files, never by splitting again.
    """
    if raw is None:
        raw = load_raw()
    split_dir = Path(split_dir)
    parts = []
    for name in ("train", "test"):
        ids = pd.read_csv(split_dir / f"{name}_ids.csv")
        part = ids.merge(raw[["review_id", "text"]], on="review_id", how="left")
        parts.append(part[["review_id", "text", "label"]])
    return parts[0], parts[1]


def build_split(raw, seed=SEED, test_size=TEST_SIZE):
    """Run steps 2–5 on the raw data and return (train, test)."""
    data = select_binary(raw)
    data = remove_duplicates(data)
    data = balance_classes(data, seed=seed)
    return split_train_test(data, test_size=test_size, seed=seed)


def main():
    raw = load_raw()
    binary = select_binary(raw)
    deduplicated = remove_duplicates(binary)
    train, test = build_split(raw)
    save_split(train, test)

    print(f"Raw reviews:                  {len(raw)}")
    print(f"Negative + positive:          {len(binary)}")
    print(f"After removing duplicates:    {len(deduplicated)} "
          f"(removed {len(binary) - len(deduplicated)})")
    print(f"Train: {len(train)}  {train['label'].value_counts().sort_index().to_dict()}")
    print(f"Test:  {len(test)}  {test['label'].value_counts().sort_index().to_dict()}")
    print(f"Saved to {SPLIT_DIR}/")


if __name__ == "__main__":
    main()
