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


def select_better_next_word(current_word, bigram_matrix, word_to_index, recent_words):
    all_words = list(word_to_index.keys())
    
    if current_word in word_to_index:
        current_idx = word_to_index[current_word]
        
        if current_idx in bigram_matrix and bigram_matrix[current_idx]:
            # Get available words and their probabilities
            available_words = {}
            for next_idx, prob in bigram_matrix[current_idx].items():
                next_word = all_words[next_idx]
                
                # Avoid recent repetition
                if next_word not in recent_words:
                    available_words[next_word] = prob
                else:
                    # Reduce probability for recently used words
                    available_words[next_word] = prob * 0.1
            
            if available_words:
                # Select based on weighted probabilities
                words = list(available_words.keys())
                probs = list(available_words.values())
                total_prob = sum(probs)
                if total_prob > 0:
                    probs = [p/total_prob for p in probs]
                    return np.random.choice(words, p=probs)
    
    # Fallback to random selection
    return np.random.choice(all_words)


def should_end_here(sentence, word_count):
    # More intelligent sentence ending logic
    if word_count < 5:
        return False
    
    if word_count >= 20:
        return True
    
    # Check for natural ending patterns
    if len(sentence) >= 2:
        last_two = sentence[-2:]
        
        # Common sentence endings
        ending_patterns = [
            ('is', 'good'), ('was', 'great'), ('are', 'fine'),
            ('can', 'help'), ('will', 'work'), ('should', 'go'),
            ('have', 'done'), ('had', 'been'), ('get', 'better')
        ]
        
        if tuple(last_two) in ending_patterns:
            return np.random.random() < 0.7
    
    # Length-based probability
    if word_count >= 8:
        end_prob = min(0.2 + (word_count - 8) * 0.05, 0.8)
        return np.random.random() < end_prob
    
    return False


def generate_structured_sentence(bigram_matrix, word_to_index, min_words=5, max_words=18):
    # Common sentence starters
    starters = ['the', 'i', 'this', 'that', 'we', 'you', 'they', 'it', 'there', 'here']
    start_word = np.random.choice(starters)
    
    sentence = [start_word]
    current_word = start_word
    recent_words = []
    
    for word_count in range(max_words):
        # Add current word to recent words
        recent_words.append(current_word)
        if len(recent_words) > 5:
            recent_words.pop(0)
        
        # Select next word with better logic
        next_word = select_better_next_word(current_word, bigram_matrix, word_to_index, recent_words)
        
        # Apply basic grammar rules
        next_word = apply_basic_grammar_rules(sentence, next_word)
        
        sentence.append(next_word)
        current_word = next_word
        
        # Check if we should end the sentence
        if word_count >= min_words - 1 and should_end_here(sentence, word_count + 1):
            break
    
    return sentence


def apply_basic_grammar_rules(sentence, next_word):
    # Basic grammar improvements
    if len(sentence) >= 1:
        prev_word = sentence[-1].lower()
        
        # Avoid repetition
        if next_word.lower() == prev_word:
            # Try to find a different word
            return next_word  # For now, just return the same word
        
        # Basic subject-verb patterns
        if prev_word == 'i' and next_word in ['am', 'was', 'will', 'have', 'had', 'can', 'should']:
            return next_word
        elif prev_word == 'the' and next_word in ['man', 'woman', 'house', 'car', 'book', 'day', 'night', 'time', 'way', 'thing']:
            return next_word
        elif prev_word == 'a' and next_word in ['man', 'woman', 'house', 'car', 'book', 'day', 'night', 'time', 'way', 'thing']:
            return next_word
    
    return next_word


def generate_text(
    bigram_matrix,
    unigram_matrix,
    word_to_index,
    num_sentences=3,
    start_words=(" ", " "),
    ending_probs=None,
    comma_probs=None,
):
    sentences = []

    for sentence_num in range(num_sentences):
        # Use the new structured sentence generation
        sentence = generate_structured_sentence(bigram_matrix, word_to_index)
        
        # Apply punctuation and capitalization
        formatted_sentence = []
        for i, word in enumerate(sentence):
            # Add punctuation
            word = add_punctuation(word, i, len(sentence), comma_probs)
            
            # Capitalize appropriately
            if i == 0:
                word = capitalize_first_letter(word)
            else:
                word = capitalize_pronouns(word)
            
            formatted_sentence.append(word)
        
        # Add sentence ending
        if not formatted_sentence[-1].endswith((".", "!", "?", ",")):
            if len(formatted_sentence) <= 10:
                ending = np.random.choice([".", "!"], p=[0.8, 0.2])
            else:
                ending = np.random.choice([".", "!", "?"], p=[0.7, 0.2, 0.1])
            formatted_sentence.append(ending)
        
        sentences.append(" ".join(formatted_sentence))

    generated_text = " ".join(sentences)

    # Clean up formatting
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
