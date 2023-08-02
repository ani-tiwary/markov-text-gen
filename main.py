from scipy.sparse import csr_matrix
from bs4 import BeautifulSoup
import numpy as np
import requests
import sys


def preprocess_text(text):
    return [word.lower() for word in text.split() if word]


def generate_transition_matrix(text):
    words = list(set(text))
    num_words = len(words)
    word_to_index = {word: i for i, word in enumerate(words)}
    transitions = {}
    for i in range(len(text) - 1):
        current_idx, next_idx = word_to_index[text[i]], word_to_index[text[i + 1]]
        transitions.setdefault(current_idx, {}).setdefault(next_idx, 0)
        transitions[current_idx][next_idx] += 1
    rows, cols, values = [], [], []
    for current_idx, next_counts in transitions.items():
        total_count = sum(next_counts.values())
        for next_idx, count in next_counts.items():
            rows.append(current_idx)
            cols.append(next_idx)
            values.append(count / total_count)
    transition_matrix = csr_matrix((values, (rows, cols)), shape=(num_words, num_words))
    return transition_matrix, word_to_index


def capitalize_first_letter(sentence):
    words = sentence.split()
    words[0] = words[0].title()
    for i, word in enumerate(words):
        if word.lower() == "i":
            words[i] = "I"
    return ' '.join(words)


def capitalize_pronouns(word):
    pronoun_mapping = {"i": "I", "i'm": "I'm", "i'll": "I'll"}
    return pronoun_mapping.get(word.lower(), word)


def generate_text(transition_matrix, word_to_index, num_words=100, start_word=' '):
    text = [start_word]
    current_word = start_word
    current_idx = word_to_index.get(current_word, -1)
    while len(text) < num_words or not text[-1].endswith(('.', '!', '?')):
        probabilities = transition_matrix[current_idx]
        probabilities = probabilities / probabilities.sum()
        next_idx = np.random.choice(len(word_to_index), p=probabilities.toarray().flatten())
        next_word = list(word_to_index.keys())[next_idx]
        next_word = capitalize_pronouns(next_word)
        if text[-1].endswith(('!', '?')):
            next_word = next_word.capitalize()
        text.append(next_word)
        current_word = next_word
        current_idx = word_to_index.get(current_word, -1)
    first_sentence, *remaining_text = ' '.join(text).split('.')
    if first_sentence:
        first_sentence = capitalize_first_letter(first_sentence)
    remaining_text = [capitalize_first_letter(sentence) if sentence else sentence for sentence in remaining_text]
    generated_text = '. '.join([first_sentence, *remaining_text])
    return generated_text


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <URL>")
        return
    url = sys.argv[1]
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch the page.")
        return
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    preprocessed_text = preprocess_text(text)
    transition_matrix, word_to_index = generate_transition_matrix(preprocessed_text)
    valid_start_words = [word for word in preprocessed_text if word.strip()]
    start_word = np.random.choice(valid_start_words)
    num_words = 50
    generated_text = generate_text(transition_matrix, word_to_index, num_words=num_words, start_word=start_word)
    print("\nGenerated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()
