from bs4 import BeautifulSoup
import numpy as np
import requests
import sys
from collections import defaultdict
import math


def preprocess_text(text):
    return [word.lower() for word in text.split() if word]


def calculate_discount(discount_factor=0.75):
    """Calculate discount factor for Kneser-Ney smoothing"""
    return discount_factor


def count_ngrams(text, n):
    """Count n-grams in text"""
    ngrams = defaultdict(int)
    for i in range(len(text) - n + 1):
        ngram = tuple(text[i:i+n])
        ngrams[ngram] += 1
    return ngrams


def count_continuations(text, n):
    """Count how many different words can follow each (n-1)-gram"""
    continuations = defaultdict(int)
    seen_contexts = defaultdict(set)
    
    for i in range(len(text) - n + 1):
        context = tuple(text[i:i+n-1])
        next_word = text[i+n-1]
        seen_contexts[context].add(next_word)
    
    for context, words in seen_contexts.items():
        continuations[context] = len(words)
    
    return continuations


def kneser_ney_probability(ngram, ngram_counts, continuation_counts, 
                          discount_factor=0.75, vocab_size=None):
    """Calculate Kneser-Ney smoothed probability for an n-gram"""
    
    if len(ngram) == 1:
        
        continuation_count = continuation_counts.get(ngram, 0)
        total_continuations = sum(continuation_counts.values())
        if total_continuations == 0:
            return 1.0 / (vocab_size or 1000)  
        return continuation_count / total_continuations
    
    
    context = ngram[:-1]
    next_word = ngram[-1]
    
    
    count = ngram_counts.get(ngram, 0)
    
    
    context_count = sum(v for k, v in ngram_counts.items() if k[:-1] == context)
    
    if context_count == 0:
        
        return kneser_ney_probability(ngram[1:], ngram_counts, continuation_counts, 
                                    discount_factor, vocab_size)
    
    
    discounted_count = max(count - discount_factor, 0)
    
    
    lambda_weight = (discount_factor * continuation_counts.get(context, 0)) / context_count
    
    
    higher_order_prob = discounted_count / context_count if context_count > 0 else 0
    
    
    lower_order_prob = kneser_ney_probability(ngram[1:], ngram_counts, continuation_counts, 
                                            discount_factor, vocab_size)
    
    return higher_order_prob + lambda_weight * lower_order_prob


def generate_transition_matrix(text):
    """Generate transition matrices using Kneser-Ney smoothing"""
    words = list(set(text))
    word_to_index = {word: i for i, word in enumerate(words)}
    vocab_size = len(words)
    discount = 0.75
    
    
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)
    
    for i in range(len(text) - 1):
        bigram_counts[(text[i], text[i+1])] += 1
        unigram_counts[text[i]] += 1
    unigram_counts[text[-1]] += 1  
    
    
    bigram_matrix = {}
    
    for (w1, w2), count in bigram_counts.items():
        if w1 in word_to_index and w2 in word_to_index:
            w1_idx = word_to_index[w1]
            w2_idx = word_to_index[w2]
            
            if w1_idx not in bigram_matrix:
                bigram_matrix[w1_idx] = {}
            
            
            context_count = unigram_counts[w1]
            discounted_count = max(count - discount, 0)
            higher_order_prob = discounted_count / context_count
            
            
            lambda_weight = discount / context_count
            unigram_prob = unigram_counts[w2] / sum(unigram_counts.values())
            
            prob = higher_order_prob + lambda_weight * unigram_prob
            bigram_matrix[w1_idx][w2_idx] = prob
    
    
    for context, probs in bigram_matrix.items():
        total = sum(probs.values())
        if total > 0:
            bigram_matrix[context] = {k: v/total for k, v in probs.items()}
    
    
    unigram_matrix = {}
    total_words = sum(unigram_counts.values())
    
    for word, count in unigram_counts.items():
        word_idx = word_to_index[word]
        unigram_matrix[word_idx] = {word_idx: count / total_words}
    
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
    """Generate text using Kneser-Ney smoothed probabilities"""
    all_words = list(word_to_index.keys())
    text = []
    
    
    if len(start_words) >= 2:
        text = list(start_words)
        current_idx = word_to_index.get(start_words[1], 0)
    else:
        
        start_idx = np.random.choice(len(all_words))
        text.append(all_words[start_idx])
        current_idx = start_idx
    
    
    recent_words = []
    
    while len(text) < num_words or not text[-1].endswith(('.', '!', '?')):
        
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
                
                if current_idx in bigram_matrix and len(bigram_matrix[current_idx]) > 1:
                    
                    available_probs = {k: v for k, v in bigram_matrix[current_idx].items() 
                                     if all_words[k] != next_word}
                    if available_probs:
                        probs = list(available_probs.values())
                        indices = list(available_probs.keys())
                        if probs:  
                            probs = np.array(probs) / np.sum(probs)  
                            next_idx = np.random.choice(indices, p=probs)
                            next_word = all_words[next_idx]
        
        
        next_word = capitalize_pronouns(next_word)
        if text and text[-1].endswith(('!', '?')):
            next_word = next_word.capitalize()
        elif text and text[-1].endswith(('.',)):
            next_word = next_word if next_word.startswith("'") else capitalize_first_letter(next_word)
        
        text.append(next_word)
        current_idx = next_idx
    
    generated_text = ' '.join(text)
    first_sentence, *remaining_text = generated_text.split('.')
    if first_sentence:
        first_sentence = capitalize_first_letter(first_sentence)
    remaining_text = [capitalize_first_letter(sentence) if sentence else sentence for sentence in remaining_text]
    generated_text = '. '.join([first_sentence, *remaining_text])
    return generated_text


def calculate_perplexity(text, bigram_matrix, unigram_matrix, word_to_index):
    """Calculate perplexity of the model on given text"""
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
            elif current_idx in unigram_matrix and next_idx in unigram_matrix[current_idx]:
                prob = unigram_matrix[current_idx][next_idx]
            else:
                
                vocab_size = len(word_to_index)
                prob = 1.0 / vocab_size
            
            if prob > 0:
                total_log_prob += math.log(prob)
                total_words += 1
    
    if total_words == 0:
        return float('inf')
    
    avg_log_prob = total_log_prob / total_words
    perplexity = math.exp(-avg_log_prob)
    return perplexity


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
    print(f"Processing {len(preprocessed_text)} words with {len(set(preprocessed_text))} unique words...")
    
    bigram_matrix, unigram_matrix, word_to_index = generate_transition_matrix(preprocessed_text)
    
    
    perplexity = calculate_perplexity(preprocessed_text, bigram_matrix, unigram_matrix, word_to_index)
    print(f"Model perplexity: {perplexity:.2f}")
    
    valid_start_words = [word for word in preprocessed_text if word.strip()]
    start_word_idx = np.random.choice(len(valid_start_words) - 1)
    start_words = (valid_start_words[start_word_idx], valid_start_words[start_word_idx + 1])
    num_words = 50
    generated_text = generate_text(bigram_matrix, unigram_matrix, word_to_index, num_words=num_words, start_words=start_words)
    print("\nGenerated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()