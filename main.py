import numpy as np
def preprocess_text(text):
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    text = ' '.join(text.split())
    return text.lower()
def generate_transition_matrix(text):
    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    num_chars = len(alphabet)
    transition_matrix = np.zeros((num_chars, num_chars), dtype=float)
    for i in range(len(text) - 1):
        current_char = text[i]
        next_char = text[i + 1]
        if current_char in alphabet and next_char in alphabet:
            current_idx = alphabet.index(current_char)
            next_idx = alphabet.index(next_char)
            transition_matrix[current_idx, next_idx] += 1
    column_sums = np.sum(transition_matrix, axis=0)
    transition_matrix = np.divide(transition_matrix, column_sums, where=column_sums != 0)
    return transition_matrix
def normalize_transition_matrix(transition_matrix):
    row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
    return transition_matrix / row_sums
def generate_text(transition_matrix, alphabet, num_chars=1000, start_char=' '):
    text = start_char
    current_char = start_char
    for i in range(num_chars):
        current_idx = alphabet.index(current_char)
        probabilities = transition_matrix[current_idx]
        next_idx = np.random.choice(len(alphabet), p=probabilities)
        next_char = alphabet[next_idx]
        text += next_char
        current_char = next_char
    return text
def main():
    input_file = 'macbeth.txt'
    with open(input_file, 'r') as file:
        text = file.read()
    preprocessed_text = preprocess_text(text)
    transition_matrix = generate_transition_matrix(preprocessed_text)
    transition_matrix = normalize_transition_matrix(transition_matrix)
    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    np.savetxt('normalized_transition_matrix.csv', transition_matrix, delimiter=',')
    generated_text = generate_text(transition_matrix, alphabet, num_chars=200, start_char=' ')
    print("\nGenerated Text:")
    print(generated_text)
if __name__ == "__main__":
    main()
