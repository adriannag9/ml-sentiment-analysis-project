"""Tests for Polish review preprocessing."""

import pytest

from src.preprocess import clean_text


def test_preserves_polish_letters():
    text = "Ą Ć Ę Ł Ń Ó Ś Ź Ż"

    assert clean_text(text) == [
        "ą", "ć", "ę", "ł", "ń", "ó", "ś", "ź", "ż"
    ]


def test_converts_text_to_lowercase():
    assert clean_text("ŚWIETNY Film") == ["świetny", "film"]


def test_replaces_punctuation_digits_and_symbols_with_spaces():
    text = "Film,nuda! Ocena: 2/10; koszt 20_zł 🙂"

    assert clean_text(text) == [
        "film", "nuda", "ocena", "koszt", "zł"
    ]


def test_collapses_whitespace():
    text = "  świetny \n\n film\t bardzo   dobry  "

    assert clean_text(text) == [
        "świetny", "film", "bardzo", "dobry"
    ]


def test_preserves_negation_and_single_letter_words():
    assert clean_text("A film nie jest zły i nudny") == [
        "a", "film", "nie", "jest", "zły", "i", "nudny"
    ]


def test_preserves_repeated_words():
    assert clean_text("film film nuda") == [
        "film", "film", "nuda"
    ]


@pytest.mark.parametrize("text", ["", " \t\n ", "123 !!!"])
def test_returns_empty_list_when_no_words_remain(text):
    assert clean_text(text) == []


def test_normalizes_decomposed_polish_characters():
    text = "Z\u0307O\u0301ŁW"

    assert clean_text(text) == ["żółw"]


@pytest.mark.parametrize("text", [None, 123, 3.14, ["film"]])
def test_rejects_non_string_input(text):
    with pytest.raises(TypeError, match="text must be a string"):
        clean_text(text)