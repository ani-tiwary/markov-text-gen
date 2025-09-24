from collections import defaultdict


def calculate_discount(discount_factor=0.75):
    return discount_factor


def kneser_ney_probability(
    ngram, ngram_counts, continuation_counts, discount_factor=0.75, vocab_size=None
):
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
        return kneser_ney_probability(
            ngram[1:], ngram_counts, continuation_counts, discount_factor, vocab_size
        )

    discounted_count = max(count - discount_factor, 0)

    lambda_weight = (
        discount_factor * continuation_counts.get(context, 0)
    ) / context_count

    higher_order_prob = discounted_count / context_count if context_count > 0 else 0

    lower_order_prob = kneser_ney_probability(
        ngram[1:], ngram_counts, continuation_counts, discount_factor, vocab_size
    )

    return higher_order_prob + lambda_weight * lower_order_prob


def generate_transition_matrix(text):
    words = list(set(text))
    word_to_index = {word: i for i, word in enumerate(words)}
    vocab_size = len(words)
    discount = 0.75

    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for i in range(len(text) - 1):
        bigram_counts[(text[i], text[i + 1])] += 1
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
            bigram_matrix[context] = {k: v / total for k, v in probs.items()}

    unigram_matrix = {}
    total_words = sum(unigram_counts.values())

    for word, count in unigram_counts.items():
        word_idx = word_to_index[word]
        unigram_matrix[word_idx] = {word_idx: count / total_words}

    return bigram_matrix, unigram_matrix, word_to_index


def generate_trigram_model(text):
    words = list(set(text))
    word_to_index = {word: i for i, word in enumerate(words)}
    vocab_size = len(words)
    
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)
    
    # Count trigrams, bigrams, and unigrams
    for i in range(len(text) - 2):
        trigram = (text[i], text[i + 1], text[i + 2])
        bigram = (text[i], text[i + 1])
        trigram_counts[trigram] += 1
        bigram_counts[bigram] += 1
        unigram_counts[text[i]] += 1
    
    # Add remaining unigrams
    for i in range(len(text) - 2, len(text)):
        unigram_counts[text[i]] += 1
    
    # Build trigram matrix
    trigram_matrix = {}
    for (w1, w2, w3), count in trigram_counts.items():
        if w1 in word_to_index and w2 in word_to_index and w3 in word_to_index:
            w1_idx = word_to_index[w1]
            w2_idx = word_to_index[w2]
            w3_idx = word_to_index[w3]
            
            if w1_idx not in trigram_matrix:
                trigram_matrix[w1_idx] = {}
            if w2_idx not in trigram_matrix[w1_idx]:
                trigram_matrix[w1_idx][w2_idx] = {}
            
            # Calculate probability
            context_count = bigram_counts.get((w1, w2), 0)
            if context_count > 0:
                prob = count / context_count
                trigram_matrix[w1_idx][w2_idx][w3_idx] = prob
    
    # Normalize probabilities
    for w1_idx in trigram_matrix:
        for w2_idx in trigram_matrix[w1_idx]:
            total = sum(trigram_matrix[w1_idx][w2_idx].values())
            if total > 0:
                trigram_matrix[w1_idx][w2_idx] = {k: v/total for k, v in trigram_matrix[w1_idx][w2_idx].items()}
    
    return trigram_matrix, word_to_index
