import numpy as np

def preprocess_text(text):
    words = text.split()
    return [' ' if word == '' else word.lower() for word in words]

def generate_transition_matrix(text):
    words = set(text)
    num_words = len(words)
    word_to_index = {word: i for i, word in enumerate(words)}
    transition_matrix = np.zeros((num_words, num_words), dtype=float)

    for i in range(len(text) - 1):
        current_word = text[i]
        next_word = text[i + 1]
        current_idx = word_to_index[current_word]
        next_idx = word_to_index[next_word]
        transition_matrix[current_idx, next_idx] += 1

    row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    return transition_matrix, word_to_index

def generate_text(transition_matrix, word_to_index, num_words=100, start_word=' '):
    text = [start_word]
    current_word = start_word

    for i in range(num_words):
        current_idx = word_to_index[current_word]
        probabilities = transition_matrix[current_idx]
        next_idx = np.random.choice(len(word_to_index), p=probabilities)
        next_word = list(word_to_index.keys())[list(word_to_index.values()).index(next_idx)]
        text.append(next_word)
        current_word = next_word

    return ' '.join(text)

def main():
    input_file = 'macbeth.txt'
    with open(input_file, 'r') as file:
        text = file.read()
    preprocessed_text = preprocess_text(text)
    transition_matrix, word_to_index = generate_transition_matrix(preprocessed_text)
    np.savetxt('normalized_transition_matrix.csv', transition_matrix, delimiter=',')

    # Choose a random valid start word from the preprocessed text
    valid_start_words = [word for word in preprocessed_text if word.strip()]
    start_word = np.random.choice(valid_start_words)

    generated_text = generate_text(transition_matrix, word_to_index, num_words=200, start_word=start_word)
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
