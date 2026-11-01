"""Tests for the train/test split (src/split.py).

Three groups:
1. Unit tests on a tiny made-up data set: check each step on its own.
2. Tests on the committed split files in data/split/: run without the raw data.
3. Tests that need the raw data (data/raw/...): skipped if it is not downloaded.
"""

import pandas as pd
import pytest

from src.split import (
    NEGATIVE,
    POSITIVE,
    RAW_PATH,
    SPLIT_DIR,
    balance_classes,
    build_split,
    load_raw,
    load_split,
    normalize_for_dedup,
    remove_duplicates,
    select_binary,
    split_train_test,
)


def make_raw(rows):
    """Build a raw-like data frame from (text, raw_label) pairs."""
    return pd.DataFrame(
        {
            "review_id": range(len(rows)),
            "text": [text for text, _ in rows],
            "raw_label": [label for _, label in rows],
        }
    )


# 1. Unit tests -------------------------------------------------------------


def test_select_binary_keeps_only_negative_and_positive():
    raw = make_raw([("a", -1), ("b", 0), ("c", 1), ("d", 2)])
    out = select_binary(raw)
    assert list(out["text"]) == ["b", "d"]
    assert list(out["label"]) == [NEGATIVE, POSITIVE]


def test_normalize_ignores_case_and_whitespace():
    assert normalize_for_dedup("  Świetny   FILM\n") == normalize_for_dedup("świetny film")


def test_remove_duplicates_keeps_first_copy():
    raw = make_raw([("Dobry film", 2), ("dobry  film", 2), ("Nuda", 0)])
    out = remove_duplicates(select_binary(raw))
    assert list(out["review_id"]) == [0, 2]


def test_remove_duplicates_drops_text_with_conflicting_labels():
    raw = make_raw([("Taki sobie", 2), ("taki sobie", 0), ("Nuda", 0)])
    out = remove_duplicates(select_binary(raw))
    assert list(out["text"]) == ["Nuda"]


def test_balance_classes_gives_equal_counts():
    rows = [(f"pos {i}", 2) for i in range(10)] + [(f"neg {i}", 0) for i in range(4)]
    out = balance_classes(select_binary(make_raw(rows)), seed=1)
    assert out["label"].value_counts().to_dict() == {NEGATIVE: 4, POSITIVE: 4}


def small_balanced():
    rows = [(f"pos {i}", 2) for i in range(50)] + [(f"neg {i}", 0) for i in range(50)]
    return select_binary(make_raw(rows))


def test_split_has_no_review_in_both_parts():
    train, test = split_train_test(small_balanced(), test_size=0.2, seed=1)
    assert set(train["review_id"]).isdisjoint(test["review_id"])
    assert len(train) + len(test) == 100


def test_split_keeps_class_ratio():
    train, test = split_train_test(small_balanced(), test_size=0.2, seed=1)
    assert test["label"].value_counts().to_dict() == {NEGATIVE: 10, POSITIVE: 10}
    assert train["label"].value_counts().to_dict() == {NEGATIVE: 40, POSITIVE: 40}


def test_split_is_the_same_for_the_same_seed():
    _, test_a = split_train_test(small_balanced(), seed=7)
    _, test_b = split_train_test(small_balanced(), seed=7)
    assert list(test_a["review_id"]) == list(test_b["review_id"])


# 2. Committed split files --------------------------------------------------


@pytest.fixture(scope="module")
def split_ids():
    train = pd.read_csv(SPLIT_DIR / "train_ids.csv")
    test = pd.read_csv(SPLIT_DIR / "test_ids.csv")
    return train, test


def test_saved_split_has_no_id_in_both_parts(split_ids):
    train, test = split_ids
    assert set(train["review_id"]).isdisjoint(test["review_id"])


def test_saved_split_has_no_repeated_ids(split_ids):
    train, test = split_ids
    assert train["review_id"].is_unique
    assert test["review_id"].is_unique


def test_saved_split_is_balanced(split_ids):
    for part in split_ids:
        counts = part["label"].value_counts()
        assert counts[NEGATIVE] == counts[POSITIVE]


# 3. Tests on the raw data --------------------------------------------------

needs_raw = pytest.mark.skipif(
    not RAW_PATH.exists(), reason=f"raw data not downloaded ({RAW_PATH})"
)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@needs_raw
def test_no_review_text_in_both_parts(raw):
    """The main leak check: compare texts, not only ids."""
    train, test = load_split(raw)
    train_texts = set(train["text"].map(normalize_for_dedup))
    test_texts = set(test["text"].map(normalize_for_dedup))
    assert train_texts.isdisjoint(test_texts)


@needs_raw
def test_saved_labels_match_raw_data(raw):
    train, test = load_split(raw)
    both = pd.concat([train, test]).merge(raw, on="review_id")
    expected = both["raw_label"].map({0: NEGATIVE, 2: POSITIVE})
    assert (both["label"] == expected).all()


@needs_raw
def test_saved_split_can_be_reproduced(raw, split_ids):
    """Running the split again gives exactly the committed files."""
    train, test = build_split(raw)
    saved_train, saved_test = split_ids
    assert sorted(train["review_id"]) == list(saved_train["review_id"])
    assert sorted(test["review_id"]) == list(saved_test["review_id"])
