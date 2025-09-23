from collections import defaultdict
from banned_words import DEFAULT_BANNED_WORDS


def preprocess_text(text, banned_words=None):
    if banned_words is None:
        banned_words = DEFAULT_BANNED_WORDS

    words = text.split()
    filtered_words = []

    for word in words:
        clean_word = "".join(c.lower() for c in word if c.isalnum())

        if clean_word and clean_word not in banned_words:
            filtered_words.append(clean_word)

    return filtered_words


def count_ngrams(text, n):
    ngrams = defaultdict(int)
    for i in range(len(text) - n + 1):
        ngram = tuple(text[i : i + n])
        ngrams[ngram] += 1
    return ngrams


def count_continuations(text, n):
    continuations = defaultdict(int)
    seen_contexts = defaultdict(set)

    for i in range(len(text) - n + 1):
        context = tuple(text[i : i + n - 1])
        next_word = text[i + n - 1]
        seen_contexts[context].add(next_word)

    for context, words in seen_contexts.items():
        continuations[context] = len(words)

    return continuations
