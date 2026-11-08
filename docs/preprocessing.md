# Text preprocessing and bag-of-words vectorization

## Installation

Run commands from the repository root.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Text preprocessing

```python
from src.preprocess import clean_text

tokens = clean_text("Świetny film! Nie jest nudny.")
# ["świetny", "film", "nie", "jest", "nudny"]
```

`clean_text(text)` accepts one string and returns a list of tokens.

Processing steps:

1. Normalize Unicode to NFC.
2. Convert text to lowercase.
3. Replace characters outside a-z and Polish letters with spaces.
4. Split on whitespace.

Polish letters, repeated words, single-letter words, and negations
are preserved. Empty input returns an empty list.
Non-string input raises TypeError.

No stop-word removal, stemming, or lemmatization is performed.

## Bag-of-words vectorization

Pass raw review strings, not token lists.

```python
from src.vectorize import BagOfWordsVectorizer

train_texts = [
    "Świetny film, świetny aktor!",
    "Nudny film, nie polecam.",
]
test_texts = [
    "Świetny film!",
    "Kosmita.",
]

vectorizer = BagOfWordsVectorizer(min_df=1, max_features=10_000)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

feature_names = vectorizer.get_feature_names_out()
vocabulary = vectorizer.vocabulary_
```

Outputs are scipy.sparse.csr_matrix objects containing integer counts.

- Each row represents one input review.
- Each column represents one vocabulary word.
- `vocabulary_` maps words to column indices.
- Unknown words are ignored.
- Reviews containing no known words produce zero rows.
- Input review order is preserved.

## Preventing data leakage

Build the vocabulary only on training reviews.

```python
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
```

Do not call fit or fit_transform on test reviews.
Do not combine training and test reviews before fitting.

Calling fit again replaces the existing vocabulary.

## Vocabulary limits

Defaults:

- `min_df=5`: a word must occur in at least five training reviews.
- `max_features=10_000`: keep at most 10,000 words.

After document-frequency filtering, words are selected by total
occurrence count. Frequency ties are resolved alphabetically.
Selected words receive column indices in alphabetical order.

These defaults are initial settings, not validated optimal values.
Use min_df=1 for small examples.

If no words satisfy the settings, fitting raises ValueError.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -rs
```

Relevant files:

- `tests/test_preprocess.py`
- `tests/test_vectorize.py`
- `tests/test_vectorize_reference.py`

Reference tests compare vocabulary and counts with scikit-learn's
CountVectorizer using the same clean_text tokenizer.

The max_features reference case avoids a frequency tie at the
selection boundary.

Full-dataset integration remains to be checked after downloading
Filmweb+ and preparing the training/test split.