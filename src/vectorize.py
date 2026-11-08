"""Sparse bag-of-words vectorization for Polish movie reviews."""

from collections import Counter
from collections.abc import Iterable

import numpy as np
from scipy.sparse import csr_matrix

from src.preprocess import clean_text


class BagOfWordsVectorizer:
    """Build a vocabulary and count words in raw review texts.

    min_df is the minimum number of documents containing a word.
    max_features limits vocabulary size by total word frequency.
    Frequency ties are resolved alphabetically.

    The vocabulary must be fitted on training documents only.
    """

    def __init__(
        self,
        min_df: int = 5,
        max_features: int | None = 10_000,
    ) -> None:
        if type(min_df) is not int or min_df < 1:
            raise ValueError("min_df must be a positive integer")

        if max_features is not None:
            if type(max_features) is not int or max_features < 1:
                raise ValueError(
                    "max_features must be a positive integer or None"
                )

        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_: dict[str, int] | None = None

    @staticmethod
    def _validate_documents(documents: Iterable[str]) -> list[str]:
        if isinstance(documents, (str, bytes)):
            raise TypeError(
                "documents must be an iterable of strings, not one string"
            )

        texts = list(documents)

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Each document must be a string")

        return texts

    def fit(self, documents: Iterable[str]) -> "BagOfWordsVectorizer":
        """Build a new vocabulary from training documents."""
        texts = self._validate_documents(documents)

        term_frequency: Counter[str] = Counter()
        document_frequency: Counter[str] = Counter()

        for text in texts:
            tokens = clean_text(text)
            term_frequency.update(tokens)
            document_frequency.update(set(tokens))

        candidates = [
            word
            for word in term_frequency
            if document_frequency[word] >= self.min_df
        ]

        ranked_words = sorted(
            candidates,
            key=lambda word: (-term_frequency[word], word),
        )

        if self.max_features is not None:
            ranked_words = ranked_words[:self.max_features]

        if not ranked_words:
            raise ValueError(
                "Empty vocabulary: no words remain. "
                "Check training texts or lower min_df."
            )

        selected_words = sorted(ranked_words)

        self.vocabulary_ = {
            word: index
            for index, word in enumerate(selected_words)
        }

        return self

    def transform(self, documents: Iterable[str]) -> csr_matrix:
        """Count known words without modifying the fitted vocabulary."""
        if self.vocabulary_ is None:
            raise RuntimeError("Call fit before transform")

        texts = self._validate_documents(documents)
        rows: list[int] = []
        columns: list[int] = []
        values: list[int] = []

        for row_index, text in enumerate(texts):
            counts = Counter(clean_text(text))

            for word, count in counts.items():
                column_index = self.vocabulary_.get(word)

                if column_index is not None:
                    rows.append(row_index)
                    columns.append(column_index)
                    values.append(count)

        matrix = csr_matrix(
            (values, (rows, columns)),
            shape=(len(texts), len(self.vocabulary_)),
            dtype=np.int64,
        )
        matrix.sort_indices()

        return matrix

    def fit_transform(self, documents: Iterable[str]) -> csr_matrix:
        """Build the vocabulary and transform the same documents."""
        texts = self._validate_documents(documents)
        self.fit(texts)

        return self.transform(texts)

    def get_feature_names_out(self) -> np.ndarray:
        """Return words in matrix-column order."""
        if self.vocabulary_ is None:
            raise RuntimeError("Call fit before requesting feature names")

        return np.array(list(self.vocabulary_), dtype=object)