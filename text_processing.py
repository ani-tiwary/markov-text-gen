from collections import defaultdict
import re
from banned_words import DEFAULT_BANNED_WORDS


def preprocess_text(text, banned_words=None):
    if banned_words is None:
        banned_words = DEFAULT_BANNED_WORDS

    # Split by sentences first to preserve structure
    sentences = re.split(r'[.!?]+', text)
    all_words = []
    
    for sentence in sentences:
        words = sentence.split()
        sentence_words = []
        
        for word in words:
            # Keep apostrophes and hyphens for contractions
            clean_word = ''.join(c.lower() for c in word if c.isalnum() or c in "'-")
            
            if clean_word and clean_word not in banned_words:
                sentence_words.append(clean_word)
        
        if len(sentence_words) >= 3:  # Only keep substantial sentences
            all_words.extend(sentence_words)
            all_words.append('<SENTENCE_END>')  # Mark sentence boundaries
    
    return all_words


def filter_nonsensical_combinations(words):
    # Remove obviously bad combinations
    bad_patterns = [
        ('i', 'love', 'i'),  # "I love I"
        ('the', 'the'),      # "the the"
        ('a', 'a'),          # "a a"
        ('is', 'is'),        # "is is"
        ('was', 'was'),      # "was was"
        ('are', 'are'),      # "are are"
        ('and', 'and'),      # "and and"
        ('or', 'or'),        # "or or"
        ('but', 'but'),      # "but but"
    ]
    
    filtered = []
    for i in range(len(words)):
        # Check for bad patterns
        is_bad = False
        for pattern in bad_patterns:
            if len(pattern) == 2:
                if i < len(words) - 1 and (words[i], words[i + 1]) == pattern:
                    is_bad = True
                    break
            elif len(pattern) == 3:
                if i < len(words) - 2 and (words[i], words[i + 1], words[i + 2]) == pattern:
                    is_bad = True
                    break
        
        if not is_bad:
            filtered.append(words[i])
    
    return filtered


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
