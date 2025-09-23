import numpy as np
import math
import re
from collections import defaultdict


def analyze_sentence_patterns(text):
    sentence_endings = defaultdict(int)
    comma_words = defaultdict(int)
    total_sentences = 0

    sentences = re.split(r"[.!?]+", text)

    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) < 3:
            continue

        total_sentences += 1

        for i, word in enumerate(words):
            clean_word = word.lower().strip(".,!?;:")
            if clean_word:
                if i >= len(words) - 3:
                    sentence_endings[clean_word] += 1

                if i < len(words) - 2:
                    if clean_word in [
                        "however",
                        "therefore",
                        "meanwhile",
                        "furthermore",
                        "moreover",
                        "nevertheless",
                        "and",
                        "but",
                        "or",
                        "so",
                    ]:
                        comma_words[clean_word] += 1

    ending_probs = {}
    for word, count in sentence_endings.items():
        if count >= 2:
            ending_probs[word] = min(count / total_sentences, 0.6)

    comma_probs = {}
    for word, count in comma_words.items():
        if count >= 2:
            comma_probs[word] = min(count / total_sentences, 0.4)

    return ending_probs, comma_probs


def capitalize_first_letter(sentence):
    words = sentence.split()
    words[0] = words[0].title()
    for i, word in enumerate(words[1:], start=1):
        if word.lower() == "i":
            words[i] = "I"
        elif word.startswith("'"):
            if len(word) > 1:
                words[i] = "'" + word[1].lower() + word[2:]
    return " ".join(words)


def capitalize_pronouns(word):
    pronoun_mapping = {"i": "I", "i'm": "I'm", "i'll": "I'll"}
    return pronoun_mapping.get(word.lower(), word)


def should_end_sentence(words, current_word, word_count, ending_probs=None):
    if word_count >= 8:
        end_probability = min(0.3 + (word_count - 8) * 0.05, 0.8)
        if np.random.random() < end_probability:
            return True

    if word_count >= 25:
        return True

    if ending_probs and current_word.lower() in ending_probs and word_count >= 6:
        if np.random.random() < ending_probs[current_word.lower()]:
            return True
    elif word_count >= 6:
        basic_endings = [
            "and",
            "but",
            "so",
            "then",
            "now",
            "here",
            "there",
            "this",
            "that",
        ]
        if current_word.lower() in basic_endings:
            if np.random.random() < 0.3:
                return True

    return False


def add_punctuation(word, position_in_sentence, sentence_length, comma_probs=None):
    if (
        comma_probs
        and word.lower() in comma_probs
        and position_in_sentence < sentence_length - 2
    ):
        if np.random.random() < comma_probs[word.lower()]:
            return word + ","
    else:
        comma_words = [
            "however",
            "therefore",
            "meanwhile",
            "furthermore",
            "moreover",
            "nevertheless",
        ]
        if word.lower() in comma_words and position_in_sentence < sentence_length - 2:
            if np.random.random() < 0.6:
                return word + ","

    if word.lower() in ["and", "but", "or", "so"] and position_in_sentence > 2:
        if np.random.random() < 0.3:
            return "," + word

    return word


def generate_text(
    bigram_matrix,
    unigram_matrix,
    word_to_index,
    num_sentences=3,
    start_words=(" ", " "),
    ending_probs=None,
    comma_probs=None,
):
    all_words = list(word_to_index.keys())
    sentences = []

    for sentence_num in range(num_sentences):
        text = []

        if sentence_num == 0 and len(start_words) >= 2:
            text = list(start_words)
            current_idx = word_to_index.get(start_words[1], 0)
        else:
            start_idx = np.random.choice(len(all_words))
            text.append(all_words[start_idx])
            current_idx = start_idx

        recent_words = []
        max_iterations = 50
        iterations = 0
        word_count = len(text)

        while iterations < max_iterations:
            iterations += 1

            if current_idx in bigram_matrix and bigram_matrix[current_idx]:
                probs = list(bigram_matrix[current_idx].values())
                indices = list(bigram_matrix[current_idx].keys())
                next_idx = np.random.choice(indices, p=probs)
            else:
                if current_idx in unigram_matrix and unigram_matrix[current_idx]:
                    probs = list(unigram_matrix[current_idx].values())
                    indices = list(unigram_matrix[current_idx].keys())
                    next_idx = np.random.choice(indices, p=probs)
                else:
                    next_idx = np.random.choice(len(all_words))

            next_word = all_words[next_idx]

            recent_words.append(next_word)
            if len(recent_words) > 3:
                recent_words.pop(0)
                if recent_words.count(next_word) > 1:
                    if (
                        current_idx in bigram_matrix
                        and len(bigram_matrix[current_idx]) > 1
                    ):
                        available_probs = {
                            k: v
                            for k, v in bigram_matrix[current_idx].items()
                            if all_words[k] != next_word
                        }
                        if available_probs:
                            probs = list(available_probs.values())
                            indices = list(available_probs.keys())
                            if probs:
                                probs = np.array(probs) / np.sum(probs)
                                next_idx = np.random.choice(indices, p=probs)
                                next_word = all_words[next_idx]

            next_word = add_punctuation(next_word, word_count, 20, comma_probs)

            next_word = capitalize_pronouns(next_word)
            if text and text[-1].endswith(("!", "?")):
                next_word = next_word.capitalize()
            elif text and text[-1].endswith((".",)):
                next_word = (
                    next_word
                    if next_word.startswith("'")
                    else capitalize_first_letter(next_word)
                )
            elif word_count == 0:
                next_word = capitalize_first_letter(next_word)

            text.append(next_word)
            current_idx = next_idx
            word_count += 1

            if should_end_sentence(text, next_word, word_count, ending_probs):
                break

        if not text[-1].endswith((".", "!", "?", ",")):
            if word_count <= 10:
                ending = np.random.choice([".", "!"], p=[0.8, 0.2])
            else:
                ending = np.random.choice([".", "!", "?"], p=[0.7, 0.2, 0.1])
            text.append(ending)

        sentences.append(" ".join(text))

    generated_text = " ".join(sentences)

    generated_text = re.sub(r"\s+", " ", generated_text)
    generated_text = re.sub(r"([.!?])\s*([.!?])", r"\1", generated_text)
    generated_text = re.sub(r",\s*,", ",", generated_text)

    return generated_text


def calculate_perplexity(text, bigram_matrix, unigram_matrix, word_to_index):
    total_log_prob = 0.0
    total_words = 0

    for i in range(len(text) - 1):
        current_word = text[i]
        next_word = text[i + 1]

        if current_word in word_to_index and next_word in word_to_index:
            current_idx = word_to_index[current_word]
            next_idx = word_to_index[next_word]

            prob = 0.0
            if current_idx in bigram_matrix and next_idx in bigram_matrix[current_idx]:
                prob = bigram_matrix[current_idx][next_idx]
            elif (
                current_idx in unigram_matrix
                and next_idx in unigram_matrix[current_idx]
            ):
                prob = unigram_matrix[current_idx][next_idx]
            else:
                vocab_size = len(word_to_index)
                prob = 1.0 / vocab_size

            if prob > 0:
                total_log_prob += math.log(prob)
                total_words += 1

    if total_words == 0:
        return float("inf")

    avg_log_prob = total_log_prob / total_words
    perplexity = math.exp(-avg_log_prob)
    return perplexity
