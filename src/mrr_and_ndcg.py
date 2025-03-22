import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


class MRR_NDCG:
    """
    A class to compute Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).
    """

    def compute_cosine_distances(self, doc_term_matrix):
        """
        Compute the pairwise cosine distances for the given document-term matrix.

        Args:
            doc_term_matrix: A 2D array or DataFrame representing the document-term matrix.

        Returns:
            A 2D array of cosine distances.
        """
        return cosine_distances(pd.DataFrame(doc_term_matrix))

    def MRR(self, doc_term_matrix, labels):
        """
        Compute the Mean Reciprocal Rank (MRR) for the given document-term matrix and labels.

        Args:
            doc_term_matrix: A 2D array or DataFrame representing the document-term matrix.
            labels: A list of labels corresponding to the documents.

        Returns:
            The Mean Reciprocal Rank (MRR) as a float.
        """
        distances = self.compute_cosine_distances(doc_term_matrix)
        reciprocal_ranks = []

        for i in tqdm(range(len(distances)), desc="Calculating MRR"):
            # Collect distances and labels for all other documents
            other_distances = []
            other_labels = []
            for j in range(len(distances)):
                if i != j:
                    other_distances.append(distances[i][j])
                    other_labels.append(labels[j])

            # Create a DataFrame of distances and labels, and sort by distance
            ranked_df = pd.DataFrame(
                list(zip(other_distances, other_labels)), columns=["distance", "label"]
            ).sort_values("distance")

            # Find the rank of the first relevant document
            for rank, label in enumerate(ranked_df["label"], start=1):
                if labels[i] == label:
                    reciprocal_ranks.append(1 / rank)
                    break

        return round(np.mean(reciprocal_ranks), 3)

    def NDCG(self, doc_term_matrix, labels):
        """
        Compute the Normalized Discounted Cumulative Gain (NDCG) for the given document-term matrix and labels.

        Args:
            doc_term_matrix: A 2D array or DataFrame representing the document-term matrix.
            labels: A list of labels corresponding to the documents.

        Returns:
            The Normalized Discounted Cumulative Gain (NDCG) as a float.
        """
        distances = self.compute_cosine_distances(doc_term_matrix)
        ndcg_scores = []

        for i in tqdm(range(len(distances)), desc="Calculating NDCG"):
            # Collect distances and labels for all other documents
            other_distances = []
            other_labels = []
            for j in range(len(distances)):
                if i != j:
                    other_distances.append(distances[i][j])
                    other_labels.append(labels[j])

            # Create a DataFrame of distances and labels, and sort by distance
            ranked_df = pd.DataFrame(
                list(zip(other_distances, other_labels)), columns=["distance", "label"]
            ).sort_values("distance")

            # Compute DCG
            dcg = 0
            for rank, label in enumerate(ranked_df["label"], start=1):
                if labels[i] == label:
                    dcg += 1 / math.log2(rank + 1)

            # Compute IDCG
            ideal_labels = sorted(other_labels, key=lambda x: x == labels[i], reverse=True)
            idcg = 0
            for rank, label in enumerate(ideal_labels, start=1):
                if labels[i] == label:
                    idcg += 1 / math.log2(rank + 1)

            # Compute NDCG
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

        return round(np.mean(ndcg_scores), 3)
