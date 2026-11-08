"""Text preprocessing for Polish movie reviews."""

import re
import unicodedata


_NON_LETTER_PATTERN = re.compile(r"[^a-ząćęłńóśźż]+")


def clean_text(text: str) -> list[str]:
    """Return lowercase tokens containing Latin and Polish letters.

    Characters outside a-z and Polish letters are replaced with spaces.
    Empty or whitespace-only input produces an empty list.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized_text = unicodedata.normalize("NFC", text)
    lowercase_text = normalized_text.lower()
    cleaned_text = _NON_LETTER_PATTERN.sub(" ", lowercase_text)

    return cleaned_text.split()