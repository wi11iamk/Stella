"""HDBSCAN clustering for pose data representations."""
import hdbscan

def perform_hdbscan_clustering(embedding, min_cluster_size=10, min_samples=5):
    """Applies HDBSCAN clustering to low-dimensional pose data representations."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusters = clusterer.fit_predict(embedding)
    return clusters, clusterer
