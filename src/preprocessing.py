import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter

# Download necessary NLTK data quietly
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

class Preprocessing:
    def preprocessing(self, requirements):
        """
        Process a list of textual requirements and return various outputs:
        - Tokenized and lemmatized corpus
        - Joined corpus as strings
        - Vocabulary list
        - Frequency distribution of vocabulary
        - Weighted average of word frequencies per document
        """
        output = []

        # Step 1: Tokenize, clean, and lemmatize the input requirements
        corpus = []
        for req in requirements:
            cleaned_text = re.sub(r'[^a-zA-Z]', ' ', req).lower()
            tokens = word_tokenize(cleaned_text)
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            corpus.append(lemmatized_tokens)
        output.append(corpus)

        # Step 2: Join tokens back into strings for each document
        joined_corpus = [' '.join(doc) for doc in corpus]
        output.append(joined_corpus)

        # Step 3: Flatten the corpus to create a vocabulary list
        vocabulary = [word for doc in corpus for word in doc]
        output.append(vocabulary)

        # Step 4: Calculate frequency distribution of the vocabulary
        vocab_freq = Counter(vocabulary)
        total_vocab_count = sum(vocab_freq.values())
        normalized_freq = {word: count / total_vocab_count for word, count in vocab_freq.items()}
        output.append(normalized_freq)

        # Step 5: Calculate weighted average of word frequencies for each document
        weighted_avg = [
            sum(normalized_freq[word] for word in doc) for doc in corpus
        ]
        output.append(weighted_avg)

        return output