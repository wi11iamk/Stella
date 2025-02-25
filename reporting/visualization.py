"""Visualization for clustering results."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_clusters(embedding, labels):
    """Plots UMAP projection with cluster assignments."""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5, alpha=0.5)
    plt.colorbar(boundaries=np.arange(len(set(labels)) + 1) - 0.5).set_ticks(np.arange(len(set(labels))))
    plt.title('UMAP Projection of Clusters')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

def plot_cluster_stability(clusterer, labels):
    """Generates a heatmap showing cluster stability scores."""
    cluster_probs = clusterer.probabilities_
    sorted_indices = np.argsort(labels)
    sorted_probs = cluster_probs[sorted_indices]

    plt.figure(figsize=(10, 5))
    sns.heatmap(sorted_probs.reshape(1, -1), cmap='coolwarm', cbar=True, xticklabels=False)
    plt.title("Cluster Stability Heatmap")
    plt.ylabel("Stability Score")
    plt.show()
