# markov-text-gen
Uses Markov chains with **Kneser-Ney smoothing** to analyze patterns in text files and generate coherent text.

## Features
- **Kneser-Ney smoothing** for better handling of unseen word combinations
- **Anti-repetition mechanisms** to avoid repetitive text generation
- **Perplexity calculation** to measure model quality
- **Smart capitalization** and punctuation handling
- **Web scraping** support for any text URL
- **Banned words filtering** to remove unwanted terms
## Dependencies
`pip install scipy beautifulsoup4 numpy requests`

## Usage
```bash
python main.py <URL>
```

### Examples
```bash
# Basic usage
python main.py https://example.com

# Using a Gutenberg text (automatically filters common Gutenberg terms)
python main.py https://www.gutenberg.org/files/11/11-h/11-h.htm
```

## Technical Details
The implementation uses:
- **Bigram and unigram models** with Kneser-Ney interpolation
- **Discount factor of 0.75** for probability smoothing
- **Fallback strategies** for unseen contexts
- **Perplexity metrics** for model evaluation
- **Word filtering** during preprocessing to remove unwanted terms

## Example Output
```
Processing 28523 words with 5521 unique words...
Model perplexity: 15.23

Generated Text:
I should be ashamed of myself, without hearing her; and a sigh: 'he won't you, will be a look like to her sister...
```

## Suggested Text Sources
The Complete Works of William Shakespeare: https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt<br>
Alice in Wonderland by Lewis Carroll: https://raw.githubusercontent.com/dakrone/corpus/master/data/alice-in-wonderland.txt<br>
Every Barack Obama Speech: https://raw.githubusercontent.com/samim23/obama-rnn/master/input.txt<br>
Every Donald Trump Speech: https://raw.githubusercontent.com/ryanmcdermott/trump-speeches/master/speeches.txt<br>
Pride And Prejudice by Jane Austen: https://gutenberg.org/cache/epub/1342/pg1342-images.html
