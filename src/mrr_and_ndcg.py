import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances

class MRR_NDCG:
    
    def MRR(self, doc_term_matrix, labels):
        """Mean Reciprocal Rank (MRR)"""
        dr = pd.DataFrame(doc_term_matrix)
        distance_matrix = cosine_distances(dr, dr)
        reciprocal_ranks = []
        
        for x in tqdm(range(len(distance_matrix))):
            distances, label_list = [], []
            rank = 1
            
            for y in range(len(distance_matrix)):
                if x != y:
                    distances.append(distance_matrix[x][y])
                    label_list.append(labels[y])
            
            sorted_df = pd.DataFrame({'distance': distances, 'label': label_list})
            sorted_df = sorted_df.sort_values('distance')
            
            for rank, label in enumerate(sorted_df['label'], start=1):
                if labels[x] == label:
                    reciprocal_ranks.append(1 / rank)
                    break
        
        return round(np.mean(reciprocal_ranks), 3)
          
    #### Normalized Discounted Cumulative Gain (NDCG) ####            
    def NDCG(self, doc_term_matrix, labels):
        dr = pd.DataFrame(doc_term_matrix)
        distance_matrix = cosine_distances(dr, dr)
        ndcg_scores = []

        for x in tqdm(range(len(distance_matrix))):
            distances = []
            label_list = []
            dcg, idcg = [], []
            rank_dcg, rank_idcg = 1, 1

            for y in range(len(distance_matrix)):
                if x != y:
                    distances.append(distance_matrix[x][y])
                    label_list.append(labels[y])

            sorted_df = pd.DataFrame(list(zip(distances, label_list)), columns=['distance', 'label'])
            sorted_df = sorted_df.sort_values('distance')

            true_labels = list(labels[:x]) + list(labels[x+1:])
            sorted_labels = sorted_df['label'].tolist()

            for b, label in enumerate(sorted_labels):
                if label == true_labels[b]:
                    dcg.append(1 / math.log2(rank_dcg + 1))
                rank_dcg += 1

            for b, label in enumerate(sorted_labels):
                if label == true_labels[b]:
                    idcg.append(1 / math.log2(rank_idcg + 1))
                    rank_idcg += 1

            ndcg_scores.append(np.sum(dcg) / np.sum(idcg) if np.sum(idcg) != 0 else 0)

        return round(np.mean(ndcg_scores), 3)
