import nltk
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import argparse
from typing import Dict, Any

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
warnings.filterwarnings("ignore")

from preprocessing import *
from automated_labeling import *
from manual_labeling import *
from utils import import_data

preprocess = Preprocessing()
automatic_labelling = AutomatedLabeling()
manual_labelling = ManualLabeling()

# Load the data after preprocessing
df, corpus, corp, all_vocab, freq, wavg, labels, namelabels = import_data()

merged_label = []
for r in labels:
    if r == 4:
        merged_label.append(0)
    else:
        merged_label.append(r)

sbert = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
s_embeddings = sbert.encode(corp)

word2vec = Word2Vec(corpus, min_count = 1,vector_size = 100,window = 5,sg = 1,epochs=30, seed = 1) # 1-> skipgram, 0-> cbow
avgword2vec = []
for x in corpus:
    avgword2vec.append(np.mean([word2vec.wv[token] for token in x if token in word2vec.wv.index_to_key],axis=0))

sroberta = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
sroberta_emb = sroberta.encode(corp)

def compare_dictionary(embeddings):

    health = []
    energy = []
    entertainment = []
    safety = []
    other = []

    for i in range(len(embeddings)):
        if namelabels[i] == 'Health':
            health.append(embeddings[i])
        elif namelabels[i] == 'Energy':
            energy.append(embeddings[i])
        elif namelabels[i] == 'Entertainment':
            entertainment.append(embeddings[i])
        elif namelabels[i] == 'Safety':
            safety.append(embeddings[i])
        elif namelabels[i] == 'Other':
            other.append(embeddings[i])

    compare = {}
    compare['health'] = np.mean(health, axis = 0)
    compare['energy'] = np.mean(energy, axis = 0)
    compare['entertainment'] = np.mean(entertainment, axis = 0)
    compare['safety'] = np.mean(safety, axis = 0)
    compare['other'] = np.mean(other, axis = 0)
    health_other = health + other
    compare['health+other'] = np.mean(health_other, axis = 0)

    return compare

embeddings_dict = {0: avgword2vec, 1: s_embeddings, 2: sroberta_emb}
clustering_dict = {0: 'kmeans', 1: 'hac'}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Clustering for Software Requirements')
    parser.add_argument('--mode', type=int, choices=[0, 1], required=True,
                       help='0 for manual labelling, 1 for automatic labelling')
    parser.add_argument('--embedding', type=int, choices=[0, 1, 2], required=True,
                       help='0: Word2Vec, 1: SBERT, 2: SRoBERTa')
    parser.add_argument('--clustering', type=int, choices=[0, 1], required=True,
                       help='0: K-means, 1: HAC')
    parser.add_argument('--merge', type=int, choices=[0, 1], required=True,
                       help='1: Merge Health and Other domains, 0: No merge')
    parser.add_argument('--clusters', type=int, required=True,
                       help='Number of clusters (2-4 if merged, 2-5 if not merged)')
    
    args = parser.parse_args()
    
    # Validate number of clusters
    max_clusters = 4 if args.merge else 5
    if not (2 <= args.clusters <= max_clusters):
        parser.error(f'Number of clusters must be between 2 and {max_clusters}')
    
    return args

def run_clustering(config: argparse.Namespace) -> None:
    """Run clustering with given configuration"""
    embedding = embeddings_dict.get(config.embedding)
    compare = compare_dictionary(embedding)
    clustering = clustering_dict.get(config.clustering)
    
    current_labels = merged_label if config.merge else labels

    try:
        if config.mode == 0:
            manual_labelling.results(
                config.clusters,
                clustering,
                current_labels,
                embedding,
                corp,
                config.merge
            )
        else:
            automatic_labelling.results(
                config.clusters,
                clustering,
                current_labels,
                embedding,
                corpus,
                compare,
                config.merge
            )
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        exit(1)

def main():
    """Main entry point"""
    args = parse_arguments()
    run_clustering(args)

if __name__ == "__main__":
    main()