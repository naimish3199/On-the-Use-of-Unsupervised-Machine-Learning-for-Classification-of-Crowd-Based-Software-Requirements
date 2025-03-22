import numpy as np
import pandas as pd

class Pretrained:
    """
    A class to compute sentence embeddings using pretrained models.
    """

    def avg_pretrained_embedding(self, model, corpus):
        """
        Computes the average sentence embedding for each sentence in a corpus using a pretrained model.

        Args:
            model: Pretrained embedding model (e.g., Word2Vec, FastText).
            corpus: List of tokenized sentences (list of lists of words).

        Returns:
            List of average embeddings for each sentence in the corpus.
        """
        embeddings = []
        skipped_words = 0

        for sentence in corpus:
            sentence_embeddings = []
            for word in sentence:
                try:
                    # Add the word embedding if it exists in the model
                    sentence_embeddings.append(model[word])
                except KeyError:
                    # Handle out-of-vocabulary (OOV) words by splitting them
                    for i in range(len(word)):
                        prefix = word[:i]
                        suffix = word[i:]
                        if prefix in model.index_to_key and suffix in model.index_to_key:
                            combined_embedding = (model[prefix] + model[suffix]) / 2
                            sentence_embeddings.append(combined_embedding)
                            break
                    skipped_words += 1

            # Compute the average embedding for the sentence
            embeddings.append(np.mean(sentence_embeddings, axis=0))

        return embeddings

    def tfidf_embedding(self, model, corpus, tfidf_array, tfidf_vocab):
        """
        Computes TF-IDF weighted sentence embeddings for each sentence in a corpus using a pretrained model.

        Args:
            model: Pretrained embedding model (e.g., Word2Vec, FastText).
            corpus: List of tokenized sentences (list of lists of words).
            tfidf_array: TF-IDF weights for each word in the corpus (2D array).
            tfidf_vocab: List of words corresponding to the TF-IDF weights.

        Returns:
            List of TF-IDF weighted embeddings for each sentence in the corpus.
        """
        embeddings = []
        skipped_words = 0

        for sentence_idx, sentence in enumerate(corpus):
            sentence_embeddings = []
            for word in sentence:
                try:
                    # Add the TF-IDF weighted word embedding if it exists in the model
                    tfidf_weight = tfidf_array[sentence_idx][tfidf_vocab.index(word)]
                    sentence_embeddings.append(model[word] * tfidf_weight)
                except (KeyError, ValueError):
                    # Handle out-of-vocabulary (OOV) words by splitting them
                    for i in range(len(word)):
                        prefix = word[:i]
                        suffix = word[i:]
                        if prefix in model.index_to_key and suffix in model.index_to_key:
                            combined_embedding = (model[prefix] + model[suffix]) / 2
                            tfidf_weight = tfidf_array[sentence_idx][tfidf_vocab.index(word)]
                            sentence_embeddings.append(combined_embedding * tfidf_weight)
                            break
                    skipped_words += 1

            # Compute the sum of TF-IDF weighted embeddings for the sentence
            embeddings.append(np.sum(sentence_embeddings, axis=0))

        return embeddings