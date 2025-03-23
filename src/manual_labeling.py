from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import numpy as np
import pandas as pd
from itertools import combinations

class ManualLabeling:
    def __init__(self):
        self.categories = None
        self.combinations = None
        self.vectorizer = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer()
        self.topic_model = None

    def generate_combinations(self, number, merged):
        """
        Generate combinations of categories based on the number of clusters and merged status.
        """
        if not merged:
            self.categories = ['health', 'energy', 'entertainment', 'safety', 'other']
        else:
            self.categories = ['health+other', 'energy', 'entertainment', 'safety']
        
        self.combinations = list(combinations(range(len(self.categories)), number))

    def perform_clustering(self, embeddings, num_clusters, method):
        """Perform clustering using specified method"""
        if method == "kmeans":
            clusterer = KMeans(n_clusters=num_clusters, random_state=42)
        elif method == "hac":
            clusterer = AgglomerativeClustering(
                n_clusters=num_clusters, 
                affinity='euclidean', 
                linkage='ward'
            )
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        return clusterer.fit_predict(embeddings)

    def extract_topics(self, texts, embeddings):
        """Extract topics using BERTopic"""
        self.topic_model = BERTopic(
            vectorizer_model=self.vectorizer,
            ctfidf_model=self.ctfidf_model,
            nr_topics=2
        )
        
        topics, _ = self.topic_model.fit_transform(texts, embeddings)
        return [word for word, _ in self.topic_model.get_topic(0)]

    def evaluate_clusters(self, true_labels, predicted_labels):
        """Calculate clustering evaluation metrics"""
        metrics = {
            'precision': precision_score(true_labels, predicted_labels, average='macro'),
            'recall': recall_score(true_labels, predicted_labels, average='macro'),
            'f1-score': f1_score(true_labels, predicted_labels, average='macro')
        }
        return metrics

    def process_combination(self, combination, labels, embeddings, corpus, num_clusters):
        """Process a single category combination"""
        # Convert embeddings to numpy array if not already
        embeddings = np.array(embeddings)
        
        # Create mask for filtering
        mask = np.array([label in combination for label in labels])
        
        # Filter data using integer indexing
        indices = np.where(mask)[0]
        filtered_embeddings = embeddings[indices]
        filtered_corpus = [corpus[i] for i in indices]
        filtered_labels = [labels[i] for i in indices]

        # Map labels to sequential integers
        label_mapping = {old: new for new, old in enumerate(combination)}
        filtered_labels = [label_mapping[label] for label in filtered_labels]

        return filtered_labels, filtered_embeddings, filtered_corpus

    def results(self, number, clustering, labels, embedd, corp, merged):
        """Main method to run the manual BERTopic classification"""
        self.generate_combinations(number, merged)
        
        for combination in self.combinations:
            category_names = [self.categories[i] for i in combination]
            print(f"\nProcessing combination: {' '.join(category_names)}")
            
            # Process combination
            filtered_labels, filtered_embeddings, filtered_corpus = self.process_combination(
                combination, labels, embedd, corp, number
            )
            
            # Perform clustering
            predictions = self.perform_clustering(filtered_embeddings, number, clustering)

            # Get cluster texts and extract topics
            cluster_texts = [[] for _ in range(number)]
            cluster_embeddings = [[] for _ in range(number)]
            for idx, cluster_id in enumerate(predictions):
                cluster_texts[cluster_id].append(filtered_corpus[idx])
                cluster_embeddings[cluster_id].append(filtered_embeddings[idx])

            # Extract and display topics
            for cluster_idx in range(number):
                if cluster_texts[cluster_idx]:
                    topics = self.extract_topics(
                        cluster_texts[cluster_idx],
                        np.array(cluster_embeddings[cluster_idx])
                    )
                    print(f"Cluster {cluster_idx} topics: {', '.join(topics)}")
            
            # Get manual labels
            print("\nAssign labels to clusters:")
            for i, cat in enumerate(category_names):
                print(f"Enter {i} for {cat}")
            
            cluster_labels = list(map(
                int, 
                input("\nEnter labels (space-separated): ").split()
            ))
            
            # Map predictions to final labels
            final_predictions = np.zeros_like(predictions)
            for cluster_idx, label in enumerate(cluster_labels):
                final_predictions[predictions == cluster_idx] = label
            
            # Evaluate and display results
            results = self.evaluate_clusters(filtered_labels, final_predictions)
            print(f"Precision: {results['precision']:.3f}, Recall: {results['recall']:.3f}, F1-Score: {results['f1-score']:.3f}")