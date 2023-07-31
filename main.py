import numpy as np
from scipy.sparse import csr_matrix


def preprocess_text(text):
    words = text.split()
    return [' ' if word == '' else word.lower() for word in words]


def generate_transition_matrix(text):
    words = set(text)
    num_words = len(words)
    word_to_index = {word: i for i, word in enumerate(words)}
    transitions = {}
    for i in range(len(text) - 1):
        current_word = text[i]
        next_word = text[i + 1]
        current_idx = word_to_index[current_word]
        next_idx = word_to_index[next_word]
        if current_idx not in transitions:
            transitions[current_idx] = {}
        if next_idx not in transitions[current_idx]:
            transitions[current_idx][next_idx] = 0
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
    for i in range(len(words)):
        if words[i].lower() == "i":
            words[i] = "I"
    return ' '.join(words)


def capitalize_pronouns(word):
    if word.lower() == "i":
        return "I"
    elif word.lower() == "i'm":
        return "I'm"
    elif word.lower() == "i'll":
        return "I'll"
    else:
        return word


def generate_text(transition_matrix, word_to_index, num_words=100, start_word=' '):
    text = [start_word]
    current_word = start_word
    current_idx = word_to_index.get(current_word, -1)
    while True:
        probabilities = transition_matrix[current_idx]
        probabilities = probabilities / probabilities.sum()
        next_idx = np.random.choice(len(word_to_index), p=probabilities.toarray().flatten())
        next_word = list(word_to_index.keys())[list(word_to_index.values()).index(next_idx)]
        next_word = capitalize_pronouns(next_word)
        next_word = capitalize_pronouns(next_word)
        text.append(next_word)
        current_word = next_word
        current_idx = word_to_index.get(current_word, -1)
        if len(text) >= num_words:
            if text[-1].endswith(('.', '!', '?')):
                break
            text.pop()
    first_sentence = ' '.join(text).split('.')
    if first_sentence:
        first_sentence[0] = capitalize_first_letter(first_sentence[0])
    generated_text = '.'.join(first_sentence)
    for i in range(1, len(text)):
        text[0] = capitalize_first_letter(text[0])
        if text[i - 1].endswith(('.', '!', '?')):
            text[i] = capitalize_first_letter(text[i])
    generated_text = ' '.join(text[:num_words])
    return generated_text


def main():
    input_file = 'macbeth.txt'
    with open(input_file, 'r') as file:
        text = file.read()
    preprocessed_text = preprocess_text(text)
    transition_matrix, word_to_index = generate_transition_matrix(preprocessed_text)
    valid_start_words = [word for word in preprocessed_text if word.strip()]
    start_word = np.random.choice(valid_start_words)
    num_words = 100
    generated_text = generate_text(transition_matrix, word_to_index, num_words=num_words, start_word=start_word)
    print("\nGenerated Text:")
    print(generated_text)
    generated_words = generated_text.split()


if __name__ == "__main__":
    main()
