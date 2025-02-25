"""UMAP dimensionality reduction for pose data representations."""
import umap

def reduce_dimensionality(features, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """Applies UMAP to reduce high-dimensional pose data embeddings."""
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    embedding = reducer.fit_transform(features)
    return embedding, reducer
