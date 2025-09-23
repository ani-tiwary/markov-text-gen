from bs4 import BeautifulSoup
import numpy as np
import requests
import sys
from text_processing import preprocess_text
from markov_model import generate_transition_matrix
from text_generator import generate_text, calculate_perplexity, analyze_sentence_patterns


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
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    preprocessed_text = preprocess_text(text)
    print(
        f"Processing {len(preprocessed_text)} words with {len(set(preprocessed_text))} unique words..."
    )

    bigram_matrix, unigram_matrix, word_to_index = generate_transition_matrix(
        preprocessed_text
    )

    # Analyze sentence patterns for dynamic ending and punctuation
    original_text = ' '.join(preprocessed_text)
    ending_probs, comma_probs = analyze_sentence_patterns(original_text)

    perplexity = calculate_perplexity(
        preprocessed_text, bigram_matrix, unigram_matrix, word_to_index
    )
    print(f"Model perplexity: {perplexity:.2f}")

    valid_start_words = [word for word in preprocessed_text if word.strip()]
    if len(valid_start_words) < 2:
        print("Error: Not enough words to generate text (need at least 2 words)")
        return

    start_word_idx = np.random.choice(len(valid_start_words) - 1)
    start_words = (
        valid_start_words[start_word_idx],
        valid_start_words[start_word_idx + 1],
    )
    num_sentences = 3
    print(f"Starting text generation with {len(word_to_index)} unique words...")
    generated_text = generate_text(
        bigram_matrix,
        unigram_matrix,
        word_to_index,
        num_sentences=num_sentences,
        start_words=start_words,
        ending_probs=ending_probs,
        comma_probs=comma_probs,
    )
    print("\nGenerated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()