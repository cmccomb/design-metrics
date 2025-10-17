"""Keyword extraction utilities inspired by the RAKE algorithm."""

from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "and",
    "of",
    "the",
    "to",
    "in",
    "for",
    "a",
    "an",
    "on",
    "with",
}

PHRASE_DELIMITERS = re.compile(r"[,.!?;:\n]\s*")
WORD_SPLIT = re.compile(r"[^A-Za-z0-9+\-#]")


def _candidate_phrases(text: str) -> list[list[str]]:
    phrases: list[list[str]] = []
    for sentence in PHRASE_DELIMITERS.split(text.lower()):
        words = [word for word in WORD_SPLIT.split(sentence) if word]
        phrase: list[str] = []
        for word in words:
            if word in STOPWORDS:
                if phrase:
                    phrases.append(phrase)
                    phrase = []
            else:
                phrase.append(word)
        if phrase:
            phrases.append(phrase)
    return phrases


def rake_keywords(text: str, *, top_k: int = 10) -> list[tuple[str, float]]:
    """Extract ranked keyword phrases from *text*.

    Args:
        text: Input document.
        top_k: Number of phrases to return.

    Returns:
        A list of ``(phrase, score)`` tuples ordered by score descending.
    """

    phrases = _candidate_phrases(text)
    word_freq: Counter[str] = Counter()
    word_degree: Counter[str] = Counter()

    for phrase in phrases:
        unique_words = set(phrase)
        degree = len(phrase) - 1
        for word in unique_words:
            word_freq[word] += 1
            word_degree[word] += degree

    word_scores = {
        word: (word_degree[word] + freq) / freq for word, freq in word_freq.items()
    }

    phrase_scores = [
        (" ".join(phrase), sum(word_scores[word] for word in phrase))
        for phrase in phrases
    ]
    phrase_scores.sort(key=lambda item: item[1], reverse=True)
    return phrase_scores[:top_k]


__all__ = ["rake_keywords"]
