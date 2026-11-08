"""Compare custom vectorization with scikit-learn."""

import pytest
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocess import clean_text
from src.vectorize import BagOfWordsVectorizer


@pytest.mark.parametrize(
    "min_df,max_features",
    [
        (1, None),
        (2, None),
        (1, 2),
    ],
)
def test_matches_sklearn_counts(min_df, max_features):
    train_texts = [
        "Świetny film film film! I dobry aktor.",
        "Dobry film, dobry aktor.",
        "Film nudny, nie polecam.",
        "Nudny film. Nie polecam!",
        "",
    ]
    test_texts = [
        "Świetny film film!",
        "Nie polecam, nudny.",
        "Kosmita galaktyka.",
        "",
    ]

    custom = BagOfWordsVectorizer(
        min_df=min_df,
        max_features=max_features,
    )
    reference = CountVectorizer(
        tokenizer=clean_text,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
    )

    custom_train = custom.fit_transform(train_texts)
    reference_train = reference.fit_transform(train_texts)

    custom_test = custom.transform(test_texts)
    reference_test = reference.transform(test_texts)

    assert (
        custom.get_feature_names_out().tolist()
        == reference.get_feature_names_out().tolist()
    )

    assert custom_train.shape == reference_train.shape
    assert custom_test.shape == reference_test.shape

    assert (custom_train != reference_train).nnz == 0
    assert (custom_test != reference_test).nnz == 0

    assert "kosmita" not in custom.vocabulary_
    assert "galaktyka" not in custom.vocabulary_