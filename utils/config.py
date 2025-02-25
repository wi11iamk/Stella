"""Configuration settings for Stella pipeline."""

def get_config():
    """Returns global configuration settings."""
    return {
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "hdbscan_min_cluster_size": 10,
        "hdbscan_min_samples": 5,
        "batch_size": 32,
        "epochs": 10,
        "logging_level": "INFO",
        "log_file": "stella.log"
    }
