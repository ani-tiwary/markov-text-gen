from bs4 import BeautifulSoup
import numpy as np
import requests
import sys


def preprocess_text(text):
    return [word.lower() for word in text.split() if word]


def generate_transition_matrix(text):
    words = list(set(text))
    word_to_index = {word: i for i, word in enumerate(words)}
    transitions = {}
    for i in range(len(text) - 2):
        current_idx = (word_to_index[text[i]], word_to_index[text[i + 1]])
        next_idx = word_to_index[text[i + 2]]
        transitions.setdefault(current_idx, {}).setdefault(next_idx, 0)
        transitions[current_idx][next_idx] += 1
    transition_matrix_dict = {}
    for current_idx, next_counts in transitions.items():
        total_count = sum(next_counts.values())
        probabilities = {next_idx: count / total_count for next_idx, count in next_counts.items()}
        transition_matrix_dict[current_idx] = probabilities
    return transition_matrix_dict, word_to_index


def capitalize_first_letter(sentence):
    words = sentence.split()
    words[0] = words[0].title()
    for i, word in enumerate(words[1:], start=1):
        if word.lower() == "i":
            words[i] = "I"
        elif word.startswith("'"):
            if len(word) > 1:
                words[i] = "'" + word[1].lower() + word[2:]
    return ' '.join(words)


def capitalize_pronouns(word):
    pronoun_mapping = {"i": "I", "i'm": "I'm", "i'll": "I'll"}
    return pronoun_mapping.get(word.lower(), word)


def generate_text(transition_matrix_dict, word_to_index, num_words=100, start_words=(' ', ' ')):
    text = list(start_words)
    current_idx = (word_to_index.get(start_words[0], -1), word_to_index.get(start_words[1], -1))

    while len(text) < num_words or not text[-1].endswith(('.', '!', '?')):
        if current_idx in transition_matrix_dict:
            probabilities = transition_matrix_dict[current_idx]
        else:
            break
        next_idx = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
        next_word = list(word_to_index.keys())[next_idx]
        next_word = capitalize_pronouns(next_word)
        if text[-1].endswith(('!', '?')):
            next_word = next_word.capitalize()
        elif text[-1].endswith(('.',)):
            next_word = next_word if next_word.startswith("'") else capitalize_first_letter(next_word)
        text.append(next_word)
        current_idx = (current_idx[1], next_idx)
    generated_text = ' '.join(text)
    first_sentence, *remaining_text = generated_text.split('.')
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
    start_word_idx = np.random.choice(len(valid_start_words) - 1)
    start_words = (valid_start_words[start_word_idx], valid_start_words[start_word_idx + 1])
    num_words = 50
    generated_text = generate_text(transition_matrix, word_to_index, num_words=num_words, start_words=start_words)
    print("\nGenerated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()
