from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from itertools import combinations


class AutomatedLabeling:
    """
    A class to perform automated labeling and clustering for various combinations of categories.
    """

    def __init__(self):
        self.categories = None
        self.combinations = None
        self.similarity_matrix = None

    def generate_combinations(self, number, merged):
        """
        Generate combinations of categories based on the number of clusters and merged status.
        """
        if not merged:
            self.categories = ['health', 'energy', 'entertainment', 'safety', 'other']
        else:
            self.categories = ['health+other', 'energy', 'entertainment', 'safety']
        
        self.combinations = list(combinations(range(len(self.categories)), number))

    def assign_clusters(self, cluster_centers, vector_list):
        """
        Assign clusters to vectors based on cosine similarity.
        """
        num_clusters = len(cluster_centers)
        num_vectors = len(vector_list)
        
        # Initialize similarity matrix
        self.similarity_matrix = np.zeros((num_clusters, num_vectors))
        
        # Compute similarity matrix
        for i, cluster in enumerate(cluster_centers):
            for j, vector in enumerate(vector_list):
                self.similarity_matrix[i, j] = cosine_similarity(
                    cluster.reshape(1, -1), 
                    vector.reshape(1, -1)
                )[0][0]
        
        # Assign vectors to clusters
        assigned_vectors = self.assign_vectors(num_clusters, num_vectors)
        return [assigned_vectors[i] for i in range(num_clusters)]

    def assign_vectors(self, num_clusters, num_vectors):
        """
        Helper method to assign vectors to clusters using similarity matrix.
        """
        assigned_vectors = {}
        while len(assigned_vectors) < min(num_clusters, num_vectors):
            # Find maximum similarity
            max_idx = np.unravel_index(
                np.argmax(self.similarity_matrix, axis=None), 
                self.similarity_matrix.shape
            )
            cluster_idx, vector_idx = max_idx
            
            # Assign cluster to vector
            assigned_vectors[cluster_idx] = vector_idx
            
            # Update similarity matrix
            self.update_similarity_matrix(cluster_idx, vector_idx)
        
        return assigned_vectors

    def update_similarity_matrix(self, cluster_idx, vector_idx):
        """
        Update similarity matrix after assignment.
        """
        self.similarity_matrix[cluster_idx, :] = -np.inf
        self.similarity_matrix[:, vector_idx] = -np.inf

    def predict_labels(self, cluster_assignments, cluster_indices, num_samples):
        """
        Predict labels based on cluster assignments.
        """
        predictions = [0] * num_samples
        for cluster_idx, assigned_idx in enumerate(cluster_assignments):
            for sample_idx in cluster_indices[cluster_idx]:
                predictions[sample_idx] = assigned_idx
        return predictions

    def results(self, number, clustering, labels, embeddings, corpus, compare, merged):
        """
        Main method to perform automated labeling.
        """
        self.generate_combinations(number, merged)
        
        for combination in self.combinations:
            print(f"\nProcessing combination: {', '.join([self.categories[i] for i in combination])}")
            
            # Filter data for current combination
            filtered_data = self.filter_data(labels, embeddings, corpus, combination)
            filtered_labels, filtered_embeddings, filtered_corpus = filtered_data
            
            # Get vectors for selected categories
            vector_list = [compare[self.categories[i]] for i in combination]
            
            # Perform clustering
            cluster_indices, cluster_centers = self.perform_clustering(
                clustering, number, filtered_embeddings
            )
            
            # Assign clusters and predict labels
            cluster_assignments = self.assign_clusters(cluster_centers, vector_list)
            predictions = self.predict_labels(
                cluster_assignments, cluster_indices, len(filtered_labels)
            )
            
            # Evaluate results
            self.evaluate_results(filtered_labels, predictions)

    def filter_data(self, labels, embeddings, corpus, combination):
        """
        Filter data based on selected combination.
        """
        filtered_labels, filtered_embeddings, filtered_corpus = [], [], []
        for i, label in enumerate(labels):
            if label in combination:
                filtered_labels.append(label)
                filtered_embeddings.append(embeddings[i])
                filtered_corpus.append(corpus[i])
        
        # Map labels to sequential integers
        label_mapping = {old: new for new, old in enumerate(combination)}
        filtered_labels = [label_mapping[label] for label in filtered_labels]
        
        return filtered_labels, filtered_embeddings, filtered_corpus

    def perform_clustering(self, clustering_method, num_clusters, embeddings):
        """
        Perform clustering on embeddings.
        """
        if clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=num_clusters, random_state=42)
        elif clustering_method == "hac":
            clusterer = AgglomerativeClustering(
                n_clusters=num_clusters, 
                affinity="euclidean", 
                linkage="ward"
            )
        else:
            raise ValueError("Invalid clustering method. Use 'kmeans' or 'hac'.")
        
        # Fit and predict clusters
        predictions = clusterer.fit_predict(embeddings)
        
        # Group indices by cluster
        cluster_indices = [[] for _ in range(num_clusters)]
        for idx, cluster in enumerate(predictions):
            cluster_indices[cluster].append(idx)
        
        # Compute cluster centers
        cluster_centers = []
        for indices in cluster_indices:
            center = np.mean([embeddings[i] for i in indices], axis=0)
            cluster_centers.append(center)
        
        return cluster_indices, cluster_centers

    def evaluate_results(self, true_labels, predicted_labels):
        """
        Evaluate clustering results.
        """
        metrics = {
            "Precision": precision_score(true_labels, predicted_labels, average="macro"),
            "Recall": recall_score(true_labels, predicted_labels, average="macro"),
            "F1-Score": f1_score(true_labels, predicted_labels, average="macro")
        }
        
        results = pd.DataFrame([{k: round(v, 3) for k, v in metrics.items()}])
        print(results)
        print('\n')
