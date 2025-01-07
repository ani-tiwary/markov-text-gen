from bs4 import BeautifulSoup
import numpy as np
import requests
import sys
from collections import defaultdict


def preprocess_text(text):
    return [word.lower() for word in text.split() if word]


def generate_transition_matrix(text):
    words = list(set(text))
    word_to_index = {word: i for i, word in enumerate(words)}
    
    bigram_transitions = defaultdict(lambda: defaultdict(int))
    unigram_transitions = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(text) - 2):
        current_idx = (word_to_index[text[i]], word_to_index[text[i + 1]])
        next_idx = word_to_index[text[i + 2]]
        bigram_transitions[current_idx][next_idx] += 1
        
        current_single_idx = word_to_index[text[i + 1]]
        unigram_transitions[current_single_idx][next_idx] += 1
    
    bigram_matrix = {}
    unigram_matrix = {}
    
    for current_idx, next_counts in bigram_transitions.items():
        total_count = sum(next_counts.values())
        bigram_matrix[current_idx] = {
            next_idx: count / total_count 
            for next_idx, count in next_counts.items()
        }
    
    for current_idx, next_counts in unigram_transitions.items():
        total_count = sum(next_counts.values())
        unigram_matrix[current_idx] = {
            next_idx: count / total_count 
            for next_idx, count in next_counts.items()
        }
    
    return bigram_matrix, unigram_matrix, word_to_index


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


def generate_text(bigram_matrix, unigram_matrix, word_to_index, num_words=100, start_words=(' ', ' ')):
    text = list(start_words)
    current_idx = (word_to_index.get(start_words[0], -1), word_to_index.get(start_words[1], -1))
    
    recent_words = []
    consecutive_bigram_uses = 0
    max_consecutive_bigrams = 3 
    
    while len(text) < num_words or not text[-1].endswith(('.', '!', '?')):
        use_unigram = False
        
        if current_idx in bigram_matrix:
            if consecutive_bigram_uses >= max_consecutive_bigrams:
                use_unigram = True
            elif len(bigram_matrix[current_idx]) == 1:
                use_unigram = np.random.random() < 0.7
            
            if not use_unigram:
                probabilities = bigram_matrix[current_idx]
                consecutive_bigram_uses += 1
            else:
                probabilities = unigram_matrix[current_idx[1]]
                consecutive_bigram_uses = 0
        else:
            if current_idx[1] in unigram_matrix:
                probabilities = unigram_matrix[current_idx[1]]
            else:
                break
        next_idx = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
        next_word = list(word_to_index.keys())[next_idx]
        recent_words.append(next_word)
        if len(recent_words) > 5:
            recent_words.pop(0)
            if recent_words.count(next_word) > 2:
                if current_idx[1] in unigram_matrix:
                    probabilities = unigram_matrix[current_idx[1]]
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
    bigram_matrix, unigram_matrix, word_to_index = generate_transition_matrix(preprocessed_text)
    valid_start_words = [word for word in preprocessed_text if word.strip()]
    start_word_idx = np.random.choice(len(valid_start_words) - 1)
    start_words = (valid_start_words[start_word_idx], valid_start_words[start_word_idx + 1])
    num_words = 50
    generated_text = generate_text(bigram_matrix, unigram_matrix, word_to_index, num_words=num_words, start_words=start_words)
    print("\nGenerated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()