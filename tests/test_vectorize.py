"""Tests for the custom bag-of-words vectorizer."""

import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr

from src.vectorize import BagOfWordsVectorizer


def test_counts_repeated_words():
    vectorizer = BagOfWordsVectorizer(min_df=1)

    matrix = vectorizer.fit_transform(["film film nuda"])

    assert matrix[0, vectorizer.vocabulary_["film"]] == 2
    assert matrix[0, vectorizer.vocabulary_["nuda"]] == 1


def test_does_not_learn_words_from_test_documents():
    vectorizer = BagOfWordsVectorizer(min_df=1)
    vectorizer.fit(["dobry film", "nudny film"])
    original_vocabulary = vectorizer.vocabulary_.copy()

    matrix = vectorizer.transform(["film kosmita kosmita"])

    assert "kosmita" not in vectorizer.vocabulary_
    assert vectorizer.vocabulary_ == original_vocabulary
    assert matrix.sum() == 1


def test_matrix_dimensions_match_documents_and_vocabulary():
    train_texts = ["dobry film", "film nuda", "dobry"]
    test_texts = ["film", "nieznane słowo"]
    vectorizer = BagOfWordsVectorizer(min_df=1)

    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)

    assert train_matrix.shape == (3, 3)
    assert test_matrix.shape == (2, 3)
    assert train_matrix.shape[1] == len(vectorizer.vocabulary_)


def test_returns_sparse_integer_matrix():
    vectorizer = BagOfWordsVectorizer(min_df=1)

    matrix = vectorizer.fit_transform(["film"])

    assert isspmatrix_csr(matrix)
    assert matrix.dtype == np.dtype("int64")


def test_min_df_counts_documents_not_word_occurrences():
    vectorizer = BagOfWordsVectorizer(min_df=2)

    vectorizer.fit(["film film film nuda", "nuda"])

    assert vectorizer.vocabulary_ == {"nuda": 0}


def test_max_features_selects_most_frequent_words():
    vectorizer = BagOfWordsVectorizer(min_df=1, max_features=2)

    vectorizer.fit(["film film film nuda nuda aktor"])

    assert vectorizer.get_feature_names_out().tolist() == [
        "film", "nuda"
    ]


def test_frequency_ties_are_resolved_alphabetically():
    vectorizer = BagOfWordsVectorizer(min_df=1, max_features=2)

    vectorizer.fit(["c b a"])

    assert vectorizer.get_feature_names_out().tolist() == ["a", "b"]


def test_preserves_rows_for_empty_and_unknown_documents():
    vectorizer = BagOfWordsVectorizer(min_df=1)
    vectorizer.fit(["film"])

    matrix = vectorizer.transform(["", "123 !!!", "kosmita"])

    assert matrix.shape == (3, 1)
    assert matrix.nnz == 0


def test_transform_accepts_empty_document_collection():
    vectorizer = BagOfWordsVectorizer(min_df=1)
    vectorizer.fit(["film"])

    matrix = vectorizer.transform([])

    assert matrix.shape == (0, 1)


def test_transform_requires_fit():
    vectorizer = BagOfWordsVectorizer()

    with pytest.raises(RuntimeError, match="Call fit"):
        vectorizer.transform(["film"])


def test_fit_rejects_empty_vocabulary():
    vectorizer = BagOfWordsVectorizer(min_df=2)

    with pytest.raises(ValueError, match="Empty vocabulary"):
        vectorizer.fit(["film"])


def test_fit_transform_accepts_generator():
    vectorizer = BagOfWordsVectorizer(min_df=1)
    documents = (text for text in ["film film", "nuda"])

    matrix = vectorizer.fit_transform(documents)

    assert matrix.shape == (2, 2)
    assert matrix.sum() == 3


def test_rejects_single_string_instead_of_document_collection():
    vectorizer = BagOfWordsVectorizer(min_df=1)

    with pytest.raises(TypeError, match="not one string"):
        vectorizer.fit("dobry film")


def test_refit_replaces_old_vocabulary():
    vectorizer = BagOfWordsVectorizer(min_df=1)
    vectorizer.fit(["film"])

    vectorizer.fit(["nuda"])

    assert vectorizer.vocabulary_ == {"nuda": 0}


@pytest.mark.parametrize(
    "parameters",
    [
        {"min_df": 0},
        {"min_df": 1.5},
        {"max_features": 0},
        {"max_features": -1},
    ],
)
def test_rejects_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        BagOfWordsVectorizer(**parameters)