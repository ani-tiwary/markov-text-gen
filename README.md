# markov-text-gen
Uses Markov chains with **Kneser-Ney smoothing** to analyze patterns in text files and generate coherent text.

## Features
- **Kneser-Ney smoothing** for better handling of unseen word combinations
- **Anti-repetition mechanisms** to avoid repetitive text generation
- **Perplexity calculation** to measure model quality
- **Smart capitalization** and punctuation handling
- **Web scraping** support for any text URL

## Dependencies
`pip install scipy beautifulsoup4 numpy requests`

## Usage
`python main.py exampleurl.com`

## Technical Details
The implementation uses:
- **Bigram and unigram models** with Kneser-Ney interpolation
- **Discount factor of 0.75** for probability smoothing
- **Fallback strategies** for unseen contexts
- **Perplexity metrics** for model evaluation

## Example Output
```
Processing 29455 words with 5575 unique words...
Model perplexity: 16.56

Generated Text:
I should be ashamed of project gutenberg-tm work, without hearing her; and a sigh: 'he won't you, will be a look like to her sister...
```

## Suggested Text Sources
The Complete Works of William Shakespeare: https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt<br>
Alice in Wonderland by Lewis Carroll: https://raw.githubusercontent.com/dakrone/corpus/master/data/alice-in-wonderland.txt<br>
Every Barack Obama Speech: https://raw.githubusercontent.com/samim23/obama-rnn/master/input.txt<br>
Every Donald Trump Speech: https://raw.githubusercontent.com/ryanmcdermott/trump-speeches/master/speeches.txt
