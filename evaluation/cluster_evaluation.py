"""Evaluation metrics for clustering results."""
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from scipy.spatial.distance import cdist

def dunn_index(embedding, labels):
    """Computes the Dunn Index for clustering quality evaluation."""
    unique_clusters = np.unique(labels[labels >= 0])
    if len(unique_clusters) < 2:
        return -1  # Dunn Index is undefined for a single cluster
    
    cluster_distances = []
    for i in unique_clusters:
        for j in unique_clusters:
            if i != j:
                cluster_i = embedding[labels == i]
                cluster_j = embedding[labels == j]
                inter_cluster_dist = np.min(cdist(cluster_i, cluster_j))
                cluster_distances.append(inter_cluster_dist)
    
    intra_cluster_diameters = [np.max(cdist(embedding[labels == i], embedding[labels == i])) for i in unique_clusters]
    
    return np.min(cluster_distances) / np.max(intra_cluster_diameters)

def evaluate_clusters(embedding, labels):
    """Computes clustering evaluation metrics."""
    valid_labels = labels[labels >= 0]
    valid_embedding = embedding[labels >= 0]
    if len(np.unique(valid_labels)) > 1:
        silhouette_avg = silhouette_score(valid_embedding, valid_labels)
        db_index = davies_bouldin_score(valid_embedding, valid_labels)
        dunn = dunn_index(valid_embedding, valid_labels)
    else:
        silhouette_avg, db_index, dunn = -1, -1, -1
        print("Warning: Only one cluster detected, evaluation metrics may be meaningless.")
    return {
        "silhouette_score": silhouette_avg,
        "davies_bouldin_index": db_index,
        "dunn_index": dunn
    }
