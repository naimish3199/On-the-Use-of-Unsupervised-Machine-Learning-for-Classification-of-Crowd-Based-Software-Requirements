from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd


class ManualBertopic:
    """
    A class to perform manual clustering and topic modeling using BERTopic.
    """

    def __init__(self):
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer()
        self.topic_model = BERTopic(
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            nr_topics=2,
        )

    def generate_combinations(self, merged, number):
        """
        Generate combinations of categories based on the number of clusters and whether categories are merged.

        Args:
            merged (bool): Whether categories are merged.
            number (int): Number of clusters.

        Returns:
            tuple: A tuple containing the category names and their combinations.
        """
        if not merged:
            categories = ["health", "energy", "entertainment", "safety", "other"]
        else:
            categories = ["health+other", "energy", "entertainment", "safety"]

        combinations = {
            2: [[i, j] for i in range(len(categories)) for j in range(i + 1, len(categories))],
            3: [[i, j, k] for i in range(len(categories)) for j in range(i + 1, len(categories)) for k in range(j + 1, len(categories))],
            4: [[i, j, k, l] for i in range(len(categories)) for j in range(i + 1, len(categories)) for k in range(j + 1, len(categories)) for l in range(k + 1, len(categories))],
            5: [[i, j, k, l, m] for i in range(len(categories)) for j in range(i + 1, len(categories)) for k in range(j + 1, len(categories)) for l in range(k + 1, len(categories)) for m in range(l + 1, len(categories))],
        }

        return categories, combinations.get(number, [])

    def filter_data(self, labels, embeddings, corpus, selected_indices):
        """
        Filter data based on selected category indices.

        Args:
            labels (list): Original labels.
            embeddings (list): Original embeddings.
            corpus (list): Original corpus.
            selected_indices (list): Selected category indices.

        Returns:
            tuple: Filtered labels, embeddings, and corpus.
        """
        filtered_labels, filtered_embeddings, filtered_corpus = [], [], []

        for i in range(len(labels)):
            if labels[i] in selected_indices:
                filtered_labels.append(labels[i])
                filtered_embeddings.append(embeddings[i])
                filtered_corpus.append(corpus[i])

        # Map labels to sequential integers
        label_mapping = {index: idx for idx, index in enumerate(selected_indices)}
        filtered_labels = [label_mapping[label] for label in filtered_labels]

        return filtered_labels, filtered_embeddings, filtered_corpus

    def perform_clustering(self, clustering_method, num_clusters, embeddings):
        """
        Perform clustering using the specified method.

        Args:
            clustering_method (str): Clustering method ('kmeans' or 'hac').
            num_clusters (int): Number of clusters.
            embeddings (list): Embeddings to cluster.

        Returns:
            list: Predicted cluster labels.
        """
        if clustering_method == "kmeans":
            model = KMeans(n_clusters=num_clusters, random_state=42)
            return model.fit_predict(embeddings)
        elif clustering_method == "hac":
            model = AgglomerativeClustering(n_clusters=num_clusters, affinity="euclidean", linkage="ward")
            return model.fit_predict(embeddings)
        else:
            raise ValueError("Invalid clustering method. Choose 'kmeans' or 'hac'.")

    def display_topics(self, clusters, embeddings):
        """
        Display topics for each cluster using BERTopic.

        Args:
            clusters (list): List of clusters.
            embeddings (list): Embeddings for each cluster.
        """
        for cluster_idx, cluster_data in enumerate(clusters):
            cluster_embeddings = np.array([embeddings[i] for i in cluster_data])
            topics, _ = self.topic_model.fit_transform(cluster_data, cluster_embeddings)
            topic_words = [word for word, _ in self.topic_model.get_topic(0)]
            print(f"Cluster {cluster_idx} -> {', '.join(topic_words)}")

    def evaluate_clustering(self, true_labels, predicted_labels):
        """
        Evaluate clustering performance using precision, recall, and F1-score.

        Args:
            true_labels (list): True labels.
            predicted_labels (list): Predicted labels.

        Returns:
            dict: Precision, recall, and F1-score.
        """
        precision = round(precision_score(true_labels, predicted_labels, average="macro"), 3)
        recall = round(recall_score(true_labels, predicted_labels, average="macro"), 3)
        f1 = round(f1_score(true_labels, predicted_labels, average="macro"), 3)
        return {"precision": precision, "recall": recall, "f1-score": f1}

    def results(self, number, clustering, labels, embeddings, corpus, merged):
        """
        Main method to perform clustering and topic modeling.

        Args:
            number (int): Number of clusters.
            clustering (str): Clustering method ('kmeans' or 'hac').
            labels (list): Original labels.
            embeddings (list): Original embeddings.
            corpus (list): Original corpus.
            merged (bool): Whether categories are merged.
        """
        categories, combinations = self.generate_combinations(merged, number)

        for combination in combinations:
            print(f"Processing combination: {', '.join([categories[i] for i in combination])}\n")

            # Filter data for the current combination
            filtered_labels, filtered_embeddings, filtered_corpus = self.filter_data(labels, embeddings, corpus, combination)

            # Perform clustering
            predicted_labels = self.perform_clustering(clustering, number, filtered_embeddings)

            # Group data by clusters
            clusters = [[] for _ in range(number)]
            for idx, cluster_label in enumerate(predicted_labels):
                clusters[cluster_label].append(filtered_corpus[idx])

            # Display topics for each cluster
            self.display_topics(clusters, filtered_embeddings)

            # Evaluate clustering performance
            evaluation_metrics = self.evaluate_clustering(filtered_labels, predicted_labels)
            print(pd.DataFrame([evaluation_metrics]))
            print("\n")